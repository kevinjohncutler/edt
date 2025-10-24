/* Multi-Label Anisotropic Euclidean Distance Transform 3D
 *
 * edt, edtsq - compute the euclidean distance transform 
 *     on a single or multi-labeled image all at once.
 *     boolean images are faster.
 *
 * binary_edt, binary_edtsq: Compute the EDT on a binary image
 *     for all input data types. Multiple labels are not handled
 *     but it's faster.
 *
 * Author: William Silversmith
 * Affiliation: Seung Lab, Princeton Neuroscience Insitute
 * Date: July 2018
 */

#ifndef EDT_H
#define EDT_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>
#include "threadpool.h"


// Portable prefetch macro (no-op if unavailable)
#if defined(__GNUC__) || defined(__clang__)
#  define EDT_PREFETCH(addr) __builtin_prefetch((addr), 0, 1)
#elif defined(_MSC_VER)
#  include <xmmintrin.h>
#  define EDT_PREFETCH(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#else
#  define EDT_PREFETCH(addr) do {} while(0)
#endif

// ND tuning knobs moved into pyedt namespace


// The pyedt namespace contains the primary implementation,
// but users will probably want to use the edt namespace (bottom)
// as the function sigs are a bit cleaner.
// pyedt names are underscored to prevent namespace collisions
// in the Cython wrapper.

namespace pyedt {
    
#define sq(x) (static_cast<float>(x) * static_cast<float>(x))

// ND tuning knobs
static size_t ND_TILE = 8;              // lines per inner tile
static size_t ND_PREFETCH_STEP = 0;     // lookahead lines for prefetch (0 to disable)
static size_t ND_CHUNKS_PER_THREAD = 1; // chunks per thread

inline void nd_set_tuning(size_t tile, size_t prefetch_step, size_t chunks_per_thread) {
  if (tile > 0) ND_TILE = tile;
  if (prefetch_step > 0) ND_PREFETCH_STEP = prefetch_step;
  if (chunks_per_thread > 0) ND_CHUNKS_PER_THREAD = chunks_per_thread;
}

inline void tofinite(float *f, const size_t voxels) {
  for (size_t i = 0; i < voxels; i++) {
    if (f[i] == std::numeric_limits<float>::infinity()) {
      f[i] = std::numeric_limits<float>::max() - 1;
    }
  }
}

inline void toinfinite(float *f, const size_t voxels) {
  for (size_t i = 0; i < voxels; i++) {
    if (f[i] >= std::numeric_limits<float>::max() - 1) {
      f[i] = std::numeric_limits<float>::infinity();
    }
  }
}

#include "loop.hpp"

namespace nd_dispatch = nd_internal;

/* 1D Euclidean Distance Transform for Multiple Segids
 *
 * Map a row of segids to a euclidean distance transform.
 * Zero is considered a universal boundary as are differing
 * segids. Segments touching the boundary are mapped to 1.
 *
 * T* segids: 1d array of (un)signed integers
 * *d: write destination, equal sized array as *segids
 * n: size of segids, d
 * stride: typically 1, but can be used on a 
 *    multi dimensional array, in which case it is nx, nx*ny, etc
 * anisotropy: physical distance of each voxel
 *
 * Writes output to *d
 */
template <typename T>
void squared_edt_1d_multi_seg(
    T* segids, float *d, const int n, 
    const long int stride, const float anistropy,
    const bool black_border=false
  ) {

  long int i;

  T working_segid = segids[0];

  if (black_border) {
    d[0] = static_cast<float>(working_segid != 0) * anistropy; // 0 or 1
  }
  else {
    d[0] = working_segid == 0 ? 0 : std::numeric_limits<float>::infinity();
  }

  for (i = stride; i < n * stride; i += stride) {
    if (segids[i] == 0) {
      d[i] = 0.0;
    }
    else if (segids[i] == working_segid) {
      d[i] = d[i - stride] + anistropy;
    }
    else {
      d[i] = anistropy;
      d[i - stride] = static_cast<float>(segids[i - stride] != 0) * anistropy;
      working_segid = segids[i];
    }
  }

  long int min_bound = 0;
  if (black_border) {
    d[n - stride] = static_cast<float>(segids[n - stride] != 0) * anistropy;
    min_bound = stride;
  }

  for (i = (n - 2) * stride; i >= min_bound; i -= stride) {
    d[i] = std::fminf(d[i], d[i + stride] + anistropy);
  }

  for (i = 0; i < n * stride; i += stride) {
    d[i] *= d[i];
  }
}

/* 1D Euclidean Distance Transform based on:
 * 
 * http://cs.brown.edu/people/pfelzens/dt/
 * 
 * Felzenszwalb and Huttenlocher. 
 * Distance Transforms of Sampled Functions.
 * Theory of Computing, Volume 8. p415-428. 
 * (Sept. 2012) doi: 10.4086/toc.2012.v008a019
 *
 * Essentially, the distance function can be 
 * modeled as the lower envelope of parabolas
 * that spring mainly from edges of the shape
 * you want to transform. The array is scanned
 * to find the parabolas, then a second scan
 * writes the correct values.
 *
 * O(N) time complexity.
 *
 * I (wms) make a few modifications for our use case
 * of executing a euclidean distance transform on
 * a 3D anisotropic image that contains many segments
 * (many binary images). This way we do it correctly
 * without running EDT > 100x in a 512^3 chunk.
 *
 * The first modification is to apply an envelope 
 * over the entire volume by defining two additional
 * vertices just off the ends at x=-1 and x=n. This
 * avoids needing to create a black border around the
 * volume (and saves 6s^2 additional memory).
 *
 * The second, which at first appeared to be important for
 * optimization, but after reusing memory appeared less important,
 * is to avoid the division operation in computing the intersection
 * point. I describe this manipulation in the code below.
 *
 * I make a third modification in squared_edt_1d_parabolic_multi_seg
 * to enable multiple segments.
 *
 * Parameters:
 *   *f: the image ("sampled function" in the paper)
 *    *d: write destination, same size in voxels as *f
 *    n: number of voxels in *f
 *    stride: 1, sx, or sx*sy to handle multidimensional arrays
 *    anisotropy: e.g. (4nm, 4nm, 40nm)
 * 
 * Returns: writes distance transform of f to d
 */
void squared_edt_1d_parabolic(
    float* f,
    const long int n, 
    const long int stride, 
    const float anisotropy, 
    const bool black_border_left,
    const bool black_border_right
  ) {

  if (n == 0) {
    return;
  }

  const float w2 = anisotropy * anisotropy;

  int k = 0;
  std::unique_ptr<int[]> v(new int[n]());
  std::unique_ptr<float[]> ff(new float[n]());
  for (long int i = 0; i < n; i++) {
    ff[i] = f[i * stride];
  }
  
  std::unique_ptr<float[]> ranges(new float[n + 1]());

  ranges[0] = -std::numeric_limits<float>::infinity();
  ranges[1] = std::numeric_limits<float>::infinity();

  /* Unclear if this adds much but I certainly find it easier to get the parens right.
   *
   * Eqn: s = ( f(r) + r^2 ) - ( f(p) + p^2 ) / ( 2r - 2p )
   * 1: s = (f(r) - f(p) + (r^2 - p^2)) / 2(r-p)
   * 2: s = (f(r) - r(p) + (r+p)(r-p)) / 2(r-p) <-- can reuse r-p, replace mult w/ add
   */
  float s;
  float factor1, factor2;
  for (long int i = 1; i < n; i++) {
    factor1 = (i - v[k]) * w2;
    factor2 =  i + v[k];
    s = (ff[i] - ff[v[k]] + factor1 * factor2) / (2.0 * factor1);

    while (k > 0 && s <= ranges[k]) {
      k--;
      factor1 = (i - v[k]) * w2;
      factor2 =  i + v[k];
      s = (ff[i] - ff[v[k]] + factor1 * factor2) / (2.0 * factor1);
    }

    k++;
    v[k] = i;
    ranges[k] = s;
    ranges[k + 1] = std::numeric_limits<float>::infinity();
  }

  k = 0;
  float envelope;
  for (long int i = 0; i < n; i++) {
    while (ranges[k + 1] < i) { 
      k++;
    }

    f[i * stride] = w2 * sq(i - v[k]) + ff[v[k]];
    // Two lines below only about 3% of perf cost, thought it would be more
    // They are unnecessary if you add a black border around the image.
    if (black_border_left && black_border_right) {
      envelope = std::fminf(w2 * sq(i + 1), w2 * sq(n - i));
      f[i * stride] = std::fminf(envelope, f[i * stride]);
    }
    else if (black_border_left) {
      f[i * stride] = std::fminf(w2 * sq(i + 1), f[i * stride]);
    }
    else if (black_border_right) {
      f[i * stride] = std::fminf(w2 * sq(n - i), f[i * stride]);      
    }
  }
}

// about 5% faster
void squared_edt_1d_parabolic(
    float* f,
    const int n, 
    const long int stride, 
    const float anisotropy
  ) {

  if (n == 0) {
    return;
  }

  const float w2 = anisotropy * anisotropy;

  int k = 0;
  std::unique_ptr<int[]> v(new int[n]());
  std::unique_ptr<float[]> ff(new float[n]());
  for (long int i = 0; i < n; i++) {
    ff[i] = f[i * stride];
  }

  std::unique_ptr<float[]> ranges(new float[n + 1]());

  ranges[0] = -std::numeric_limits<float>::infinity();
  ranges[1] = std::numeric_limits<float>::infinity();

  /* Unclear if this adds much but I certainly find it easier to get the parens right.
   *
   * Eqn: s = ( f(r) + r^2 ) - ( f(p) + p^2 ) / ( 2r - 2p )
   * 1: s = (f(r) - f(p) + (r^2 - p^2)) / 2(r-p)
   * 2: s = (f(r) - r(p) + (r+p)(r-p)) / 2(r-p) <-- can reuse r-p, replace mult w/ add
   */
  float s;
  float factor1, factor2;
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
    ranges[k + 1] = std::numeric_limits<float>::infinity();
  }

  k = 0;
  float envelope;
  for (long int i = 0; i < n; i++) {
    while (ranges[k + 1] < i) { 
      k++;
    }

    f[i * stride] = w2 * sq(i - v[k]) + ff[v[k]];
    // Two lines below only about 3% of perf cost, thought it would be more
    // They are unnecessary if you add a black border around the image.
    envelope = std::fminf(w2 * sq(i + 1), w2 * sq(n - i));
    f[i * stride] = std::fminf(envelope, f[i * stride]);
  }
}

// Variant that also records the true argmin index per element (v[k]).
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
  const float w2 = anisotropy * anisotropy;

  int k = 0;
  std::unique_ptr<int[]> v(new int[n]());
  std::unique_ptr<float[]> ff(new float[n]());
  for (long int i = 0; i < n; i++) ff[i] = f[i * stride];
  std::unique_ptr<float[]> ranges(new float[n + 1]());
  ranges[0] = -std::numeric_limits<float>::infinity();
  ranges[1] = std::numeric_limits<float>::infinity();

  float s; float factor1, factor2;
  for (long int i = 1; i < n; i++) {
    factor1 = (i - v[k]) * w2;
    factor2 =  i + v[k];
    s = (ff[i] - ff[v[k]] + factor1 * factor2) / (2.0f * factor1);
    while (k > 0 && s <= ranges[k]) {
      k--;
      factor1 = (i - v[k]) * w2;
      factor2 =  i + v[k];
      s = (ff[i] - ff[v[k]] + factor1 * factor2) / (2.0f * factor1);
    }
    k++;
    v[k] = i;
    ranges[k] = s;
    ranges[k + 1] = std::numeric_limits<float>::infinity();
  }

  k = 0;
  float envelope;
  for (long int i = 0; i < n; i++) {
    while (ranges[k + 1] < i) k++;
    f[i * stride] = w2 * sq(i - v[k]) + ff[v[k]];
    arg_out[i * arg_stride] = v[k];
    if (black_border_left && black_border_right) {
      envelope = std::fminf(w2 * sq(i + 1), w2 * sq(n - i));
      f[i * stride] = std::fminf(envelope, f[i * stride]);
    } else if (black_border_left) {
      f[i * stride] = std::fminf(w2 * sq(i + 1), f[i * stride]);
    } else if (black_border_right) {
      f[i * stride] = std::fminf(w2 * sq(n - i), f[i * stride]);
    }
  }
}

inline void squared_edt_1d_parabolic_with_arg(
    float* f,
    const long int n,
    const long int stride,
    const float anisotropy,
    const bool black_border_left,
    const bool black_border_right,
    int* arg_out
  ) {
  squared_edt_1d_parabolic_with_arg_stride(
      f, n, stride, anisotropy,
      black_border_left, black_border_right,
      arg_out, stride);
}

inline void squared_edt_1d_parabolic_with_arg(
    float* f,
    const int n,
    const long int stride,
    const float anisotropy,
    int* arg_out
  ) {
  if (n == 0) return;
  const float w2 = anisotropy * anisotropy;
  int k = 0;
  std::unique_ptr<int[]> v(new int[n]());
  std::unique_ptr<float[]> ff(new float[n]());
  for (long int i = 0; i < n; i++) ff[i] = f[i * stride];
  std::unique_ptr<float[]> ranges(new float[n + 1]());
  ranges[0] = -std::numeric_limits<float>::infinity();
  ranges[1] = std::numeric_limits<float>::infinity();
  float s; float factor1, factor2;
  for (long int i = 1; i < n; i++) {
    factor1 = (i - v[k]) * w2;
    factor2 =  i + v[k];
    s = (ff[i] - ff[v[k]] + factor1 * factor2) / (2.0f * factor1);
    while (k > 0 && s <= ranges[k]) {
      k--;
      factor1 = (i - v[k]) * w2;
      factor2 =  i + v[k];
      s = (ff[i] - ff[v[k]] + factor1 * factor2) / (2.0f * factor1);
    }
    k++;
    v[k] = i;
    ranges[k] = s;
    ranges[k + 1] = std::numeric_limits<float>::infinity();
  }
  k = 0;
  for (long int i = 0; i < n; i++) {
    while (ranges[k + 1] < i) k++;
    f[i * stride] = w2 * sq(i - v[k]) + ff[v[k]];
    arg_out[i * stride] = v[k];
    float envelope = std::fminf(w2 * sq(i + 1), w2 * sq(n - i));
    f[i * stride] = std::fminf(envelope, f[i * stride]);
  }
}

void _squared_edt_1d_parabolic(
    float* f, 
    const int n, 
    const long int stride, 
    const float anisotropy, 
    const bool black_border_left,
    const bool black_border_right
  ) {

  if (black_border_left && black_border_right) {
    squared_edt_1d_parabolic(f, n, stride, anisotropy);
  }
  else {
    squared_edt_1d_parabolic(f, n, stride, anisotropy, black_border_left, black_border_right); 
  }
}

/* Same as squared_edt_1d_parabolic except that it handles
 * a simultaneous transform of multiple labels (like squared_edt_1d_multi_seg).
 * 
 *  Parameters:
 *    *segids: an integer labeled image where 0 is background
 *    *f: the image ("sampled function" in the paper)
 *    n: number of voxels in *f
 *    stride: 1, sx, or sx*sy to handle multidimensional arrays
 *    anisotropy: e.g. (4.0 = 4nm, 40.0 = 40nm)
 * 
 * Returns: writes squared distance transform in f
 */
// Modified: allow optional arg_out for argmin tracking (feature transform)
template <typename T>
void squared_edt_1d_parabolic_multi_seg(
  T* segids, float* f,
  const int n, const long int stride, const float anisotropy,
  const bool black_border=false,
  int* arg_out=nullptr
) {
  constexpr int SMALL_THRESHOLD = 8;
  const float anis_sq = anisotropy * anisotropy;

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
  T segid;
  long int last = 0;

  for (int i = 1; i < n; i++) {
    segid = segids[i * stride];
    if (segid != working_segid) {
      if (working_segid != 0) {
        const int run_len = i - last;
        const bool left_border = (black_border || last > 0);
        const bool right_border = true;
        if (arg_out == nullptr && run_len <= SMALL_THRESHOLD) {
          process_small_run(last, run_len, left_border, right_border);
        } else if (arg_out) {
          // true argmin indices (relative to run), then offset to absolute
          squared_edt_1d_parabolic_with_arg(
            f + last * stride,
            i - last, stride, anisotropy,
            (black_border || last > 0), true,
            arg_out + last * stride
          );
          for (int j = 0; j < i - last; ++j) {
            arg_out[(last + j) * stride] += last;
          }
        } else {
          _squared_edt_1d_parabolic(
            f + last * stride,
            i - last, stride, anisotropy,
            (black_border || last > 0), true
          );
        }
      }
      working_segid = segid;
      last = i;
    }
  }

  if (working_segid != 0 && last < n) {
    const int run_len = n - last;
    const bool left_border = (black_border || last > 0);
    const bool right_border = black_border;
    if (arg_out == nullptr && run_len <= SMALL_THRESHOLD) {
      process_small_run(last, run_len, left_border, right_border);
    } else if (arg_out) {
      squared_edt_1d_parabolic_with_arg(
        f + last * stride,
        n - last, stride, anisotropy,
        (black_border || last > 0), black_border,
        arg_out + last * stride
      );
      for (int j = 0; j < n - last; ++j) {
        arg_out[(last + j) * stride] += last;
      }
    } else {
      _squared_edt_1d_parabolic(
        f + last * stride,
        n - last, stride, anisotropy,
        (black_border || last > 0), black_border
      );
    }
  }
}

// Convenience wrapper to expose arg_out to Cython without changing the original name
template <typename T>
inline void squared_edt_1d_parabolic_multi_seg_with_arg(
  T* segids, float* f,
  const int n, const long int stride, const float anisotropy,
  const bool black_border, int* arg_out
) {
  squared_edt_1d_parabolic_multi_seg<T>(segids, f, n, stride, anisotropy, black_border, arg_out);
}

#include "edt_specific.hpp"

// ------------------------------
// ND threaded axis passes (for Cython ND core)
// ------------------------------

template <typename T>
inline void _nd_pass_multi(
    T* labels, float* dest,
    const size_t dims,
    const size_t* shape,
    const size_t* strides,
    const size_t ax,
    const float anis,
    const bool black_border,
    const int parallel
) {
  size_t total = 1;
  for (size_t d = 0; d < dims; ++d) total *= shape[d];
  const size_t n = shape[ax];
  if (n == 0) return;
  const size_t s = strides[ax];
  const size_t lines = total / n;

  const int threads = std::max(1, parallel);
  ThreadPool pool(threads);
  size_t chunks = std::max<size_t>(1, std::min<size_t>(lines, (size_t)threads * ND_CHUNKS_PER_THREAD));
  const size_t chunk = (lines + chunks - 1) / chunks;

  for (size_t start = 0; start < lines; start += chunk) {
    const size_t end = std::min(lines, start + chunk);
    pool.enqueue([=]() {
      for (size_t idx_line = start; idx_line < end; ++idx_line) {
        size_t base = 0;
        size_t tmp = idx_line;
        for (size_t d = 0; d < dims; ++d) {
          if (d == ax) continue;
          const size_t coord = tmp % shape[d];
          base += coord * strides[d];
          tmp /= shape[d];
        }
        squared_edt_1d_multi_seg<T>(labels + base, dest + base, (int)n, (long int)s, anis, black_border);
      }
    });
  }
  pool.join();
}

template <typename T>
inline void _nd_pass_parabolic(
    T* labels, float* dest,
    const size_t dims,
    const size_t* shape,
    const size_t* strides,
    const size_t ax,
    const float anis,
    const bool black_border,
    const int parallel
) {
  size_t total = 1;
  for (size_t d = 0; d < dims; ++d) total *= shape[d];
  const size_t n = shape[ax];
  if (n == 0) return;
  const size_t s = strides[ax];
  const size_t lines = total / n;

  const int threads = std::max(1, parallel);
  ThreadPool pool(threads);
  // Allow finer task granularity: more than one chunk per thread when helpful
  size_t chunks = std::max<size_t>(1, std::min<size_t>(lines, (size_t)threads * ND_CHUNKS_PER_THREAD));
  const size_t chunk = (lines + chunks - 1) / chunks;

  for (size_t start = 0; start < lines; start += chunk) {
    const size_t end = std::min(lines, start + chunk);
    pool.enqueue([=]() {
      for (size_t idx_line = start; idx_line < end; ++idx_line) {
        size_t base = 0;
        size_t tmp = idx_line;
        for (size_t d = 0; d < dims; ++d) {
          if (d == ax) continue;
          const size_t coord = tmp % shape[d];
          base += coord * strides[d];
          tmp /= shape[d];
        }
        squared_edt_1d_parabolic_multi_seg<T>(labels + base, dest + base, (int)n, (long int)s, anis, black_border);
      }
    });
  }
  pool.join();
}

// Threaded pass using precomputed base offsets (faster: no per-line mod/div)
template <typename T>
inline void _nd_pass_multi_bases(
    T* labels, float* dest,
    const size_t* bases, const size_t num_lines,
    const size_t n, const size_t s,
    const float anis,
    const bool black_border,
    const int parallel
) {
  if (n == 0 || num_lines == 0) return;
  const int threads = std::max(1, parallel);
  if (threads <= 1 || num_lines == 1) {
    for (size_t i = 0; i < num_lines; ++i) {
      const size_t base = bases[i];
      squared_edt_1d_multi_seg<T>(labels + base, dest + base, (int)n, (long int)s, anis, black_border);
    }
    return;
  }
  auto& pool = ::pyedt::nd_dispatch::shared_pool_for(static_cast<size_t>(threads));
  std::vector<std::future<void>> pending;
  pending.reserve(threads);
  size_t chunks = std::max<size_t>(1, std::min<size_t>(num_lines, (size_t)threads * ND_CHUNKS_PER_THREAD));
  const size_t chunk = (num_lines + chunks - 1) / chunks;
  const size_t TILE = ND_TILE;
  for (size_t start = 0; start < num_lines; start += chunk) {
    const size_t end = std::min(num_lines, start + chunk);
    pending.push_back(pool.enqueue([=]() {
      for (size_t i = start; i < end; i += TILE) {
        const size_t t_end = std::min(end, i + TILE);
        for (size_t j = i; j < t_end; ++j) {
          if (ND_PREFETCH_STEP > 0) {
            const size_t pre = j + ND_PREFETCH_STEP;
            if (pre < t_end) {
              EDT_PREFETCH(labels + bases[pre]);
            }
          }
          const size_t base = bases[j];
          squared_edt_1d_multi_seg<T>(labels + base, dest + base, (int)n, (long int)s, anis, black_border);
        }
      }
    }));
  }
  for (auto& f : pending) {
    f.get();
  }
}

template <typename T>
inline void _nd_pass_parabolic_bases(
    T* labels, float* dest,
    const size_t* bases, const size_t num_lines,
    const size_t n, const size_t s,
    const float anis,
    const bool black_border,
    const int parallel
) {
  if (n == 0 || num_lines == 0) return;
  const int threads = std::max(1, parallel);
  const size_t TILE = ND_TILE;
  if (threads <= 1 || num_lines == 1) {
    for (size_t i = 0; i < num_lines; ++i) {
      const size_t base = bases[i];
      squared_edt_1d_parabolic_multi_seg<T>(labels + base, dest + base, (int)n, (long int)s, anis, black_border);
    }
    return;
  }
  auto& pool = ::pyedt::nd_dispatch::shared_pool_for(static_cast<size_t>(threads));
  std::vector<std::future<void>> pending;
  pending.reserve(threads);
  size_t chunks = std::max<size_t>(1, std::min<size_t>(num_lines, (size_t)threads * ND_CHUNKS_PER_THREAD));
  const size_t chunk = (num_lines + chunks - 1) / chunks;
  for (size_t start = 0; start < num_lines; start += chunk) {
    const size_t end = std::min(num_lines, start + chunk);
    pending.push_back(pool.enqueue([=]() {
      for (size_t i = start; i < end; i += TILE) {
        const size_t t_end = std::min(end, i + TILE);
        for (size_t j = i; j < t_end; ++j) {
          if (ND_PREFETCH_STEP > 0) {
            const size_t pre = j + ND_PREFETCH_STEP;
            if (pre < t_end) {
              EDT_PREFETCH(labels + bases[pre]);
            }
          }
          const size_t base = bases[j];
          squared_edt_1d_parabolic_multi_seg<T>(labels + base, dest + base, (int)n, (long int)s, anis, black_border);
        }
      }
    }));
  }
  for (auto& f : pending) {
    f.get();
  }
}

template <typename T>
inline bool _nd_pass_multi_compiled(
    T* labels,
    float* dest,
    const size_t dims,
    const size_t* shape,
    const size_t* strides,
    const size_t ax,
    const float anis,
    const bool black_border,
    const int parallel) {
  if (dims == 0 || dims > 5) {
    return false;
  }
  const size_t n = shape[ax];
  if (n == 0) {
    return true;
  }
  if (parallel <= 1) {
    return false;
  }

  size_t extents_buf[5] = {0, 0, 0, 0, 0};
  size_t stride_buf[5] = {0, 0, 0, 0, 0};
  size_t m = 0;
  for (size_t i = 0; i < dims; ++i) {
    if (i == ax) continue;
    extents_buf[m] = shape[i];
    stride_buf[m] = strides[i];
    ++m;
  }

  size_t num_lines = 1;
  for (size_t i = 0; i < m; ++i) {
    if (extents_buf[i] == 0) {
      num_lines = 0;
      break;
    }
    num_lines *= extents_buf[i];
  }
  if (num_lines == 0) {
    return true;
  }

  std::vector<size_t> bases(num_lines);
  size_t idx = 0;
  auto record_base = [&](size_t base) {
    if (idx < num_lines) {
      bases[idx++] = base;
    }
  };

  switch (m) {
    case 0:
      bases[0] = 0;
      break;
    case 1:
      ::pyedt::nd_dispatch::for_each_line<1>(extents_buf, stride_buf, size_t{0}, record_base);
      break;
    case 2:
      ::pyedt::nd_dispatch::for_each_line<2>(extents_buf, stride_buf, size_t{0}, record_base);
      break;
    case 3:
      ::pyedt::nd_dispatch::for_each_line<3>(extents_buf, stride_buf, size_t{0}, record_base);
      break;
    case 4:
      ::pyedt::nd_dispatch::for_each_line<4>(extents_buf, stride_buf, size_t{0}, record_base);
      break;
    default:
      return false;
  }

  if (idx != num_lines) {
    return false;
  }

  const size_t stride_ax = strides[ax];
  _nd_pass_multi_bases<T>(
      labels,
      dest,
      bases.data(),
      num_lines,
      n,
      stride_ax,
      anis,
      black_border,
      parallel);
  return true;
}

template <typename T>
inline bool _nd_pass_parabolic_compiled(
    T* labels,
    float* dest,
    const size_t dims,
    const size_t* shape,
    const size_t* strides,
    const size_t ax,
    const float anis,
    const bool black_border,
    const int parallel) {
  if (dims == 0 || dims > 5) {
    return false;
  }
  const size_t n = shape[ax];
  if (n == 0) {
    return true;
  }
  if (parallel <= 1) {
    return false;
  }

  size_t extents_buf[5] = {0, 0, 0, 0, 0};
  size_t stride_buf[5] = {0, 0, 0, 0, 0};
  size_t m = 0;
  for (size_t i = 0; i < dims; ++i) {
    if (i == ax) continue;
    extents_buf[m] = shape[i];
    stride_buf[m] = strides[i];
    ++m;
  }

  size_t num_lines = 1;
  for (size_t i = 0; i < m; ++i) {
    if (extents_buf[i] == 0) {
      num_lines = 0;
      break;
    }
    num_lines *= extents_buf[i];
  }
  if (num_lines == 0) {
    return true;
  }

  std::vector<size_t> bases(num_lines);
  size_t idx = 0;
  auto record_base = [&](size_t base) {
    if (idx < num_lines) {
      bases[idx++] = base;
    }
  };

  switch (m) {
    case 0:
      bases[0] = 0;
      break;
    case 1:
      ::pyedt::nd_dispatch::for_each_line<1>(extents_buf, stride_buf, size_t{0}, record_base);
      break;
    case 2:
      ::pyedt::nd_dispatch::for_each_line<2>(extents_buf, stride_buf, size_t{0}, record_base);
      break;
    case 3:
      ::pyedt::nd_dispatch::for_each_line<3>(extents_buf, stride_buf, size_t{0}, record_base);
      break;
    case 4:
      ::pyedt::nd_dispatch::for_each_line<4>(extents_buf, stride_buf, size_t{0}, record_base);
      break;
    default:
      return false;
  }

  if (idx != num_lines) {
    return false;
  }

  const size_t stride_ax = strides[ax];
  _nd_pass_parabolic_bases<T>(
      labels,
      dest,
      bases.data(),
      num_lines,
      n,
      stride_ax,
      anis,
      black_border,
      parallel);
  return true;
}

// Odometer-style threaded pass (no base array; per-chunk init, then incrementally update base)
template <typename T, typename KERNEL>
inline void _nd_pass_odometer(
    T* labels, float* dest,
    const size_t dims,
    const size_t* shape,
    const size_t* strides,
    const size_t ax,
    const float anis,
    const bool black_border,
    const int parallel,
    KERNEL kernel_call
) {
  size_t total = 1;
  for (size_t d = 0; d < dims; ++d) total *= shape[d];
  const size_t n = shape[ax];
  if (n == 0) return;
  const size_t s = strides[ax];
  const size_t lines = total / n;

  // Build axis order (other axes) by increasing stride, tie-break larger extent
  const size_t m = dims - 1;
  std::vector<size_t> ord; ord.reserve(m);
  for (size_t d = 0; d < dims; ++d) if (d != ax) ord.push_back(d);
  std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b){
    if (strides[a] == strides[b]) return shape[a] > shape[b];
    return strides[a] < strides[b];
  });
  std::vector<size_t> rad(m);
  for (size_t i = 0; i < m; ++i) rad[i] = shape[ord[i]];

  const int threads = std::max(1, parallel);
  const size_t tile_size = ND_TILE > 0 ? ND_TILE : 8;

  auto process_range = [&](size_t start, size_t end) {
    size_t tmp = start;
    std::vector<size_t> idx(m, 0);
    std::vector<size_t> sim_idx(m, 0);
    std::vector<size_t> bases_tile(tile_size);
    size_t base = 0;
    for (size_t i = 0; i < m; ++i) {
      if (rad[i] > 0) {
        idx[i] = tmp % rad[i];
        tmp /= rad[i];
        base += idx[i] * strides[ord[i]];
      }
    }
    const size_t count = end - start;
    size_t cdone = 0;
    while (cdone < count) {
      const size_t tcount = std::min(tile_size, count - cdone);
      size_t sim_base = base;
      sim_idx = idx;
      for (size_t t = 0; t < tcount; ++t) {
        bases_tile[t] = sim_base;
        for (size_t i = 0; i < m; ++i) {
          ++sim_idx[i];
          sim_base += strides[ord[i]];
          if (sim_idx[i] < rad[i]) break;
          sim_base -= strides[ord[i]] * rad[i];
          sim_idx[i] = 0;
        }
      }
      for (size_t t = 0; t < tcount; ++t) {
        if (ND_PREFETCH_STEP > 0) {
          const size_t pre = t + ND_PREFETCH_STEP;
          if (pre < tcount) {
            EDT_PREFETCH(labels + bases_tile[pre]);
          }
        }
        const size_t b = bases_tile[t];
        kernel_call(labels + b, dest + b, (int)n, (long int)s, anis, black_border);
      }
      base = sim_base;
      idx.swap(sim_idx);
      cdone += tcount;
    }
  };

  if (threads <= 1) {
    process_range(0, lines);
    return;
  }

  ThreadPool& pool = ::pyedt::nd_dispatch::shared_pool_for(static_cast<size_t>(threads));
  std::vector<std::future<void>> pending;
  pending.reserve(static_cast<size_t>(threads));
  size_t chunks = std::max<size_t>(1, std::min<size_t>(lines, (size_t)threads));
  const size_t chunk = (lines + chunks - 1) / chunks;

  for (size_t start = 0; start < lines; start += chunk) {
    const size_t end = std::min(lines, start + chunk);
    pending.push_back(pool.enqueue([=]() { process_range(start, end); }));
  }
  for (auto& f : pending) {
    f.get();
  }
}

template <typename T>
inline void _nd_pass_multi_odometer(
    T* labels, float* dest,
    const size_t dims,
    const size_t* shape,
    const size_t* strides,
    const size_t ax,
    const float anis,
    const bool black_border,
    const int parallel
) {
  auto kernel = [&](T* l, float* d, int n, long s, float an, bool bb){
    squared_edt_1d_multi_seg<T>(l, d, n, s, an, bb);
  };
  _nd_pass_odometer<T>(labels, dest, dims, shape, strides, ax, anis, black_border, parallel, kernel);
}

template <typename T>
inline void _nd_pass_parabolic_odometer(
    T* labels, float* dest,
    const size_t dims,
    const size_t* shape,
    const size_t* strides,
    const size_t ax,
    const float anis,
    const bool black_border,
    const int parallel
) {
  auto kernel = [&](T* l, float* d, int n, long s, float an, bool bb){
    squared_edt_1d_parabolic_multi_seg<T>(l, d, n, s, an, bb);
  };
  _nd_pass_odometer<T>(labels, dest, dims, shape, strides, ax, anis, black_border, parallel, kernel);
}

// ------------------------------
// ND expand labels (feature transform) helpers
// ------------------------------

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
          // all seeded: arg is identity
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
  auto process_line = [&](size_t base) {
    std::vector<int> arg(n);
    std::vector<INDEX> feat_line(n);
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
  };

  if (threads <= 1 || num_lines == 1) {
    for (size_t i = 0; i < num_lines; ++i) {
      process_line(bases[i]);
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

// Label-propagation versions to avoid full feature-index arrays when only labels are needed
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

} // namespace pyedt

#endif // EDT_H

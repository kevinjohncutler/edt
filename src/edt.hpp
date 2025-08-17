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


// The pyedt namespace contains the primary implementation,
// but users will probably want to use the edt namespace (bottom)
// as the function sigs are a bit cleaner.
// pyedt names are underscored to prevent namespace collisions
// in the Cython wrapper.

namespace pyedt {
    
// --- Feature-transform forward declarations (2D/3D) ---
template <typename T>
float* _edt2dsq_features(
    T* labels, size_t sx, size_t sy,
    float wx, float wy,
    bool black_border, int parallel,
    float* output_dt, size_t* output_feat,
    int tie_mode);

template <typename T>
float* _edt3dsq_features(
    T* labels, size_t sx, size_t sy, size_t sz,
    float wx, float wy, float wz,
    bool black_border, int parallel,
    float* output_dt, size_t* output_feat,
    int tie_mode);

#define sq(x) (static_cast<float>(x) * static_cast<float>(x))

inline void tofinite(float *f, const size_t voxels) {
  for (size_t i = 0; i < voxels; i++) {
    if (f[i] == INFINITY) {
      f[i] = std::numeric_limits<float>::max() - 1;
    }
  }
}

inline void toinfinite(float *f, const size_t voxels) {
  for (size_t i = 0; i < voxels; i++) {
    if (f[i] >= std::numeric_limits<float>::max() - 1) {
      f[i] = INFINITY;
    }
  }
}

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
    d[0] = working_segid == 0 ? 0 : INFINITY;
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

  ranges[0] = -INFINITY;
  ranges[1] = +INFINITY;

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
    ranges[k + 1] = +INFINITY;
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

  ranges[0] = -INFINITY;
  ranges[1] = +INFINITY;

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
    ranges[k + 1] = +INFINITY;
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
template <typename T>
void squared_edt_1d_parabolic_multi_seg(
  T* segids, float* f,
  const int n, const long int stride, const float anisotropy,
  const bool black_border=false
) {

  T working_segid = segids[0];
  T segid;
  long int last = 0;

  for (int i = 1; i < n; i++) {
    segid = segids[i * stride];
    if (segid != working_segid) {
      if (working_segid != 0) {
        _squared_edt_1d_parabolic(
          f + last * stride, 
          i - last, stride, anisotropy,
          (black_border || last > 0), true
        );
      }
      working_segid = segid;
      last = i;
    }
  }

  if (working_segid != 0 && last < n) {
    _squared_edt_1d_parabolic(
      f + last * stride, 
      n - last, stride, anisotropy,
      (black_border || last > 0), black_border
    );
  }
}

/* Df(x,y,z) = min( wx^2 * (x-x')^2 + Df|x'(y,z) )
 *              x'                   
 * Df(y,z) = min( wy^2 * (y-y') + Df|x'y'(z) )
 *            y'
 * Df(z) = wz^2 * min( (z-z') + i(z) )
 *          z'
 * i(z) = 0   if voxel in set (f[p] == 1)
 *        inf if voxel out of set (f[p] == 0)
 *
 * In english: a 3D EDT can be accomplished by
 *    taking the x axis EDT, followed by y, followed by z.
 * 
 * The 2012 paper by Felzenszwalb and Huttenlocher describes using
 * an indicator function (above) to use their sampled function
 * concept on all three axes. This is unnecessary. The first
 * transform (x here) can be done very dumbly and cheaply using
 * the method of Rosenfeld and Pfaltz (1966) in 1D (where the L1
 * and L2 norms agree). This first pass is extremely fast and so
 * saves us about 30% in CPU time. 
 *
 * The second and third passes use the Felzenszalb and Huttenlocher's
 * method. The method uses a scan then write sequence, so we are able
 * to write to our input block, which increases cache coherency and
 * reduces memory usage.
 *
 * Parameters:
 *    *labels: an integer labeled image where 0 is background
 *    sx, sy, sz: size of the volume in voxels
 *    wx, wy, wz: physical dimensions of voxels (weights)
 *
 * Returns: writes squared distance transform of f to d
 */
template <typename T>
float* _edt3dsq(
    T* labels, 
    const size_t sx, const size_t sy, const size_t sz, 
    const float wx, const float wy, const float wz,
    const bool black_border=false, const int parallel=1,
    float* workspace=NULL
  ) {

  const size_t sxy = sx * sy;
  const size_t voxels = sz * sxy;

  if (workspace == NULL) {
    workspace = new float[sx * sy * sz]();
  }

  ThreadPool pool(parallel);

  for (size_t z = 0; z < sz; z++) {
    pool.enqueue([labels, sy, z, sx, sxy, wx, workspace, black_border](){
      for (size_t y = 0; y < sy; y++) {
        squared_edt_1d_multi_seg<T>(
          (labels + sx * y + sxy * z), 
          (workspace + sx * y + sxy * z), 
          sx, 1, wx, black_border
        ); 
      }
    });
  }

  pool.join();

  if (!black_border) {
    tofinite(workspace, voxels);
  }

  pool.start(parallel);

  for (size_t z = 0; z < sz; z++) {
    pool.enqueue([labels, sxy, z, workspace, sx, sy, wy, black_border](){
      for (size_t x = 0; x < sx; x++) {
        squared_edt_1d_parabolic_multi_seg<T>(
          (labels + x + sxy * z),
          (workspace + x + sxy * z), 
          sy, sx, wy, black_border
        );
      }
    });
  }

  pool.join();
  pool.start(parallel);

  for (size_t y = 0; y < sy; y++) {
    pool.enqueue([labels, sx, y, workspace, sz, sxy, wz, black_border](){
      for (size_t x = 0; x < sx; x++) {
        squared_edt_1d_parabolic_multi_seg<T>(
          (labels + x + sx * y), 
          (workspace + x + sx * y), 
          sz, sxy, wz, black_border
        );
      }
    });
  }

  pool.join();

  if (!black_border) {
    toinfinite(workspace, voxels);
  }

  return workspace; 
}

// skipping multi-seg logic results in a large speedup
template <typename T>
float* _binary_edt3dsq(
    T* binaryimg, 
    const size_t sx, const size_t sy, const size_t sz, 
    const float wx, const float wy, const float wz,
    const bool black_border=false, const int parallel=1, 
    float* workspace=NULL
  ) {

  const size_t sxy = sx * sy;
  const size_t voxels = sz * sxy;

  size_t x,y,z;

  if (workspace == NULL) {
    workspace = new float[sx * sy * sz]();
  }  

  ThreadPool pool(parallel);
  
  for (z = 0; z < sz; z++) {
    for (y = 0; y < sy; y++) { 
      pool.enqueue([binaryimg, sx, y, sxy, z, workspace, wx, black_border](){
        squared_edt_1d_multi_seg<T>(
          (binaryimg + sx * y + sxy * z), 
          (workspace + sx * y + sxy * z), 
          sx, 1, wx, black_border
        ); 
      });
    }
  }

  pool.join();

  if (!black_border) {
    tofinite(workspace, voxels);
  }

  pool.start(parallel);

  size_t offset;
  for (z = 0; z < sz; z++) {
    for (x = 0; x < sx; x++) {
      offset = x + sxy * z;
      for (y = 0; y < sy; y++) {
        if (workspace[offset + sx*y]) {
          break;
        }
      }

      pool.enqueue([sx, sy, y, workspace, wy, black_border, offset](){
        _squared_edt_1d_parabolic(
          (workspace + offset + sx * y),
          sy - y, sx, wy, 
          black_border || (y > 0), black_border
        );
      });
    }
  }

  pool.join();
  pool.start(parallel);

  for (y = 0; y < sy; y++) {
    for (x = 0; x < sx; x++) {
      offset = x + sx * y;
      pool.enqueue([sz, sxy, workspace, wz, black_border, offset](){
        size_t z = 0;
        for (z = 0; z < sz; z++) {
          if (workspace[offset + sxy*z]) {
            break;
          }
        }
        _squared_edt_1d_parabolic(
          (workspace + offset + sxy * z), 
          sz - z, sxy, wz, 
          black_border || (z > 0), black_border
        );
      });
    }
  }

  pool.join();

  if (!black_border) {
    toinfinite(workspace, voxels);
  }

  return workspace; 
}

// about 20% faster on binary images by skipping
// multisegment logic in parabolic
template <typename T>
float* _edt3dsq(bool* binaryimg, 
  const size_t sx, const size_t sy, const size_t sz, 
  const float wx, const float wy, const float wz, 
  const bool black_border=false, const int parallel=1, float* workspace=NULL) {

  return _binary_edt3dsq(binaryimg, sx, sy, sz, wx, wy, wz, black_border, parallel, workspace);
}

// Same as _edt3dsq, but applies square root to get
// euclidean distance.
template <typename T>
float* _edt3d(T* input, 
  const size_t sx, const size_t sy, const size_t sz, 
  const float wx, const float wy, const float wz,
  const bool black_border=false, const int parallel=1, float* workspace=NULL) {

  float* transform = _edt3dsq<T>(input, sx, sy, sz, wx, wy, wz, black_border, parallel, workspace);

  for (size_t i = 0; i < sx * sy * sz; i++) {
    transform[i] = std::sqrt(transform[i]);
  }

  return transform;
}

// skipping multi-seg logic results in a large speedup
template <typename T>
float* _binary_edt3d(
    T* input, 
    const size_t sx, const size_t sy, const size_t sz, 
    const float wx, const float wy, const float wz,
    const bool black_border=false, const int parallel=1, 
    float* workspace=NULL
  ) {

  float* transform = _binary_edt3dsq<T>(
    input, 
    sx, sy, sz, 
    wx, wy, wz, 
    black_border, parallel, 
    workspace
  );

  for (size_t i = 0; i < sx * sy * sz; i++) {
    transform[i] = std::sqrt(transform[i]);
  }

  return transform;
}

// 2D version of _edt3dsq
template <typename T>
float* _edt2dsq(
    T* input, 
    const size_t sx, const size_t sy,
    const float wx, const float wy,
    const bool black_border=false, const int parallel=1,
    float* workspace=NULL
  ) {

  const size_t voxels = sx * sy;

  if (workspace == NULL) {
    workspace = new float[voxels]();
  }

  for (size_t y = 0; y < sy; y++) { 
    squared_edt_1d_multi_seg<T>(
      (input + sx * y), (workspace + sx * y), 
      sx, 1, wx, black_border
    ); 
  }

  if (!black_border) {
    tofinite(workspace, voxels);
  }

  ThreadPool pool(parallel);

  for (size_t x = 0; x < sx; x++) {
    pool.enqueue([input, x, workspace, sy, sx, wy, black_border](){
      squared_edt_1d_parabolic_multi_seg<T>(
        (input + x), 
        (workspace + x), 
        sy, sx, wy,
        black_border
      );
    });
  }

  pool.join();

  if (!black_border) {
    toinfinite(workspace, voxels);
  }

  return workspace;
}

// skipping multi-seg logic results in a large speedup
template <typename T>
float* _binary_edt2dsq(T* binaryimg, 
  const size_t sx, const size_t sy,
  const float wx, const float wy,
  const bool black_border=false, const int parallel=1,
  float* workspace=NULL) {

  const size_t voxels = sx * sy;
  size_t x,y;

  if (workspace == NULL) {
    workspace = new float[sx * sy]();
  }

  for (y = 0; y < sy; y++) { 
    squared_edt_1d_multi_seg<T>(
      (binaryimg + sx * y), (workspace + sx * y), 
      sx, 1, wx, black_border
    ); 
  }

  if (!black_border) {
    tofinite(workspace, voxels);
  }

  ThreadPool pool(parallel);

  for (x = 0; x < sx; x++) {
    pool.enqueue([workspace, x, sx, sy, wy, black_border](){
      size_t y = 0;
      for (y = 0; y < sy; y++) {
        if (workspace[x + y * sx]) {
          break;
        }
      }

      _squared_edt_1d_parabolic(
        (workspace + x + y * sx), 
        sy - y, sx, wy,
        black_border || (y > 0), black_border
      );
    });
  }

  pool.join();

  if (!black_border) {
    toinfinite(workspace, voxels);
  }

  return workspace;
}

// skipping multi-seg logic results in a large speedup
template <typename T>
float* _binary_edt2d(T* binaryimg, 
  const size_t sx, const size_t sy,
  const float wx, const float wy,
  const bool black_border=false, const int parallel=1,
  float* output=NULL) {

  float *transform = _binary_edt2dsq(
    binaryimg, 
    sx, sy, 
    wx, wy, 
    black_border, parallel, 
    output
  );

  for (size_t i = 0; i < sx * sy; i++) {
    transform[i] = std::sqrt(transform[i]);
  }

  return transform;
}

// 2D version of _edt3dsq
template <typename T>
float* _edt2dsq(bool* binaryimg, 
  const size_t sx, const size_t sy,
  const float wx, const float wy,
  const bool black_border=false, const int parallel=1,
  float* output=NULL) {

  return _binary_edt2dsq(
    binaryimg, 
    sx, sy, 
    wx, wy, 
    black_border, parallel,
    output
  );
}

// returns euclidean distance instead of squared distance
template <typename T>
float* _edt2d(
    T* input, 
    const size_t sx, const size_t sy,
    const float wx, const float wy,
    const bool black_border=false, const int parallel=1,
    float* output=NULL
  ) {

  float* transform = _edt2dsq<T>(
    input, 
    sx, sy, 
    wx, wy, 
    black_border, parallel, 
    output
  );

  for (size_t i = 0; i < sx * sy; i++) {
    transform[i] = std::sqrt(transform[i]);
  }

  return transform;
}


// ================================
// Feature-transform implementations
// ================================
// Tie-breaking policy for equal-distance cases.
// 0 = prefer "last" site encountered (current behavior)
// 1 = prefer "first" (lexicographically smaller index)
enum { TIE_LAST = 0, TIE_FIRST = 1 };

namespace detail_ft {

// Felzenszwalb 1D lower envelope + argmin (with weight w, tie-breaking)
inline void dt1d_parabolic_with_arg(
    const float* f_in, int n, float w, int tie_mode,
    std::vector<float>& d, std::vector<int>& arg) {
  d.assign(n, std::numeric_limits<float>::infinity());
  arg.assign(n, -1);
  if (n <= 0) return;

  const float w2 = w * w;
  // Tiny bias to make tie policy effective without affecting distances
  const float bias_scale = (tie_mode == TIE_FIRST) ? +1e-7f : -1e-7f;
  auto biased_val = [&](int i, const float* fptr) {
    return fptr[i] + w2 * float(i) * float(i) + bias_scale * float(i);
  };

  // thread-local scratch to avoid alloc churn
  static thread_local std::vector<int>   v_buf;
  static thread_local std::vector<float> z_buf;
  static thread_local std::vector<float> f_buf;

  v_buf.resize(n);
  z_buf.resize(n + 1);
  f_buf.resize(n);

  for (int i = 0; i < n; ++i) f_buf[i] = f_in[i];

  int* v = v_buf.data();
  float* z = z_buf.data();
  const float* f = f_buf.data();

  int k = 0;
  v[0] = 0;
  z[0] = -INFINITY;
  z[1] = +INFINITY;

  auto inter = [&](int i, int j) -> float {
    const float fi = biased_val(i, f);
    const float fj = biased_val(j, f);
    return (fi - fj) / (2.f * w2 * float(i - j));
  };

  for (int q = 1; q < n; ++q) {
    float s = inter(q, v[k]);
    // tie policy: <= (prefer last) vs < (prefer first)
    if (tie_mode == TIE_FIRST) {
      while (k >= 0 && s < z[k]) {
        --k;
        if (k >= 0) s = inter(q, v[k]);
      }
    } else { // TIE_LAST
      while (k >= 0 && s <= z[k]) {
        --k;
        if (k >= 0) s = inter(q, v[k]);
      }
    }
    ++k;
    v[k]   = q;
    z[k]   = s;
    z[k+1] = +INFINITY;
  }

  k = 0;
  for (int x = 0; x < n; ++x) {
    while (z[k + 1] < x) ++k;
    const int p = v[k];
    d[x]   = w2 * (x - p) * (x - p) + f[p];
    arg[x] = p;
  }
}

} // namespace detail_ft


// 2D: carry x-arg through X pass, lift to (x,y) in Y pass
template <typename T>
float* _edt2dsq_features(
    T* labels, size_t sx, size_t sy,
    float wx, float wy,
    bool /*black_border*/, int parallel,
    float* output_dt, size_t* output_feat,
    int tie_mode) {

  const size_t voxels = sx * sy;
  std::unique_ptr<float[]> tmp(new float[voxels]);
  std::unique_ptr<int[]>   argx(new int[voxels]);
  float* tmp_p = tmp.get();
  int*   argx_p = argx.get();

  // --- X pass (per row), parallel over blocks of rows ---
  {
    ThreadPool pool(std::max(1, parallel));
    const size_t by = std::max<size_t>(1, sy / (size_t(4) * std::max(1, parallel)));
    for (size_t y0 = 0; y0 < sy; y0 += by) {
      const size_t y1 = std::min(sy, y0 + by);
      pool.enqueue([=]() {
        std::vector<float> drow; std::vector<int> arow; std::vector<float> f(sx);
        for (size_t y = y0; y < y1; ++y) {
          const size_t off = y * sx;
          for (size_t x = 0; x < sx; ++x) f[x] = (labels[off + x] != 0) ? 0.f : std::numeric_limits<float>::infinity();
          detail_ft::dt1d_parabolic_with_arg(f.data(), int(sx), wx, tie_mode, drow, arow);
          for (size_t x = 0; x < sx; ++x) { tmp_p[off + x] = drow[x]; argx_p[off + x] = arow[x]; }
        }
      });
    }
    pool.join();
  }

  // --- Y pass (per column), parallel over blocks of columns ---
  {
    ThreadPool pool(std::max(1, parallel));
    const size_t bx = std::max<size_t>(1, sx / (size_t(4) * std::max(1, parallel)));
    for (size_t x0 = 0; x0 < sx; x0 += bx) {
      const size_t x1 = std::min(sx, x0 + bx);
      pool.enqueue([=]() {
        std::vector<float> dcol; std::vector<int> arow; std::vector<float> g(sy); std::vector<int> axcol(sy);
        for (size_t x = x0; x < x1; ++x) {
          for (size_t y = 0; y < sy; ++y) { const size_t idx = y * sx + x; g[y] = tmp_p[idx]; axcol[y] = argx_p[idx]; }
          detail_ft::dt1d_parabolic_with_arg(g.data(), int(sy), wy, tie_mode, dcol, arow);
          for (size_t y = 0; y < sy; ++y) {
            const int r = arow[y]; const size_t idx = y * sx + x;
            output_dt[idx] = dcol[y];
            output_feat[idx] = (r >= 0 && axcol[r] >= 0) ? size_t(r) * sx + size_t(axcol[r]) : size_t(0);
          }
        }
      });
    }
    pool.join();
  }

  return output_dt;
}


// 3D: X → Y → Z, carrying (seed x, seed y), finalize with z
template <typename T>
float* _edt3dsq_features(
    T* labels, size_t sx, size_t sy, size_t sz,
    float wx, float wy, float wz,
    bool /*black_border*/, int parallel,
    float* output_dt, size_t* output_feat,
    int tie_mode) {

  const size_t sxy = sx * sy;
  const size_t voxels = sxy * sz;

  std::unique_ptr<float[]> dx(new float[voxels]);
  std::unique_ptr<int[]>   sx_seed(new int[voxels]);
  float* dx_p = dx.get();
  int*   sx_seed_p = sx_seed.get();

  // X pass (per y-row in each z-slice), chunked by blocks of y
  {
    ThreadPool pool(std::max(1, parallel));
    const size_t by = std::max<size_t>(1, sy / (size_t(4) * std::max(1, parallel)));
    for (size_t z = 0; z < sz; ++z) {
      for (size_t y0 = 0; y0 < sy; y0 += by) {
        const size_t y1 = std::min(sy, y0 + by);
        pool.enqueue([=]() {
          std::vector<float> drow; std::vector<int> arow; std::vector<float> f(sx);
          const size_t base_z = z * sxy;
          for (size_t y = y0; y < y1; ++y) {
            const size_t base = base_z + y * sx;
            for (size_t x = 0; x < sx; ++x) f[x] = (labels[base + x] != 0) ? 0.f : std::numeric_limits<float>::infinity();
            detail_ft::dt1d_parabolic_with_arg(f.data(), int(sx), wx, tie_mode, drow, arow);
            for (size_t x = 0; x < sx; ++x) { dx_p[base + x] = drow[x]; sx_seed_p[base + x] = arow[x]; }
          }
        });
      }
    }
    pool.join();
  }

  std::unique_ptr<float[]> dxy(new float[voxels]);
  std::unique_ptr<int[]>   sy_seed(new int[voxels]);
  std::unique_ptr<int[]>   sx_seed_y(new int[voxels]);
  float* dxy_p = dxy.get();
  int*   sy_seed_p = sy_seed.get();
  int*   sx_seed_y_p = sx_seed_y.get();

  // Y pass (per x-column in each z-slice), chunked by blocks of x
  {
    ThreadPool pool(std::max(1, parallel));
    const size_t bx = std::max<size_t>(1, sx / (size_t(4) * std::max(1, parallel)));
    for (size_t z = 0; z < sz; ++z) {
      for (size_t x0 = 0; x0 < sx; x0 += bx) {
        const size_t x1 = std::min(sx, x0 + bx);
        pool.enqueue([=]() {
          std::vector<float> dcol; std::vector<int> arow; std::vector<float> g(sy); std::vector<int> axcol(sy);
          for (size_t x = x0; x < x1; ++x) {
            for (size_t y = 0; y < sy; ++y) { const size_t idx = z * sxy + y * sx + x; g[y] = dx_p[idx]; axcol[y] = sx_seed_p[idx]; }
            detail_ft::dt1d_parabolic_with_arg(g.data(), int(sy), wy, tie_mode, dcol, arow);
            for (size_t y = 0; y < sy; ++y) { const size_t idx = z * sxy + y * sx + x; const int r = arow[y]; dxy_p[idx] = dcol[y]; sy_seed_p[idx] = r; sx_seed_y_p[idx] = (r >= 0 ? axcol[r] : -1); }
          }
        });
      }
    }
    pool.join();
  }

  // Z pass (per (x,y) column), chunked by blocks of columns
  {
    ThreadPool pool(std::max(1, parallel));
    const size_t bxy = std::max<size_t>(1, (sx * sy) / (size_t(4) * std::max(1, parallel)));
    const size_t cols = sx * sy;
    for (size_t c0 = 0; c0 < cols; c0 += bxy) {
      const size_t c1 = std::min(cols, c0 + bxy);
      pool.enqueue([=]() {
        std::vector<float> dline; std::vector<int> arow; std::vector<float> g(sz); std::vector<int> ayline(sz);
        for (size_t c = c0; c < c1; ++c) {
          const size_t y = c / sx; const size_t x = c % sx;
          for (size_t z = 0; z < sz; ++z) { const size_t idx = z * sxy + y * sx + x; g[z] = dxy_p[idx]; ayline[z] = sy_seed_p[idx]; }
          detail_ft::dt1d_parabolic_with_arg(g.data(), int(sz), wz, tie_mode, dline, arow);
          for (size_t z = 0; z < sz; ++z) {
            const size_t idx = z * sxy + y * sx + x; const int rz = arow[z];
            output_dt[idx] = dline[z];
            if (rz >= 0) {
              const int ry = ayline[rz];
              const int rx = (ry >= 0) ? sx_seed_y_p[rz * sxy + size_t(ry) * sx + x] : -1;
              output_feat[idx] = (rx >= 0 && ry >= 0) ? size_t(rx) + sx * (size_t(ry) + sy * size_t(rz)) : size_t(0);
            } else {
              output_feat[idx] = 0;
            }
          }
        }
      });
    }
    pool.join();
  }

  return output_dt;
}



// Should be trivial to make an N-d version
// if someone asks for it. Might simplify the interface.

} // namespace pyedt

namespace edt {

template <typename T>
float* edt(
  T* labels, 
  const int sx, const float wx, 
  const bool black_border=false) {

  float* d = new float[sx]();
  pyedt::squared_edt_1d_multi_seg(labels, d, sx, 1, wx);

  for (int i = 0; i < sx; i++) {
    d[i] = std::sqrt(d[i]);
  }

  return d;
}

template <typename T>
float* edt(
    T* labels, 
    const int sx, const int sy, 
    const float wx, const float wy,
    const bool black_border=false, const int parallel=1,
    float* output=NULL
  ) {

  return pyedt::_edt2d(labels, sx, sy, wx, wy, black_border, parallel, output);
}


template <typename T>
float* edt(
  T* labels, 
  const int sx, const int sy, const int sz, 
  const float wx, const float wy, const float wz,
  const bool black_border=false, const int parallel=1, float* output=NULL) {

  return pyedt::_edt3d(labels, sx, sy, sz, wx, wy, wz, black_border, parallel, output);
}

template <typename T>
float* binary_edt(
  T* labels, 
  const int sx, 
  const float wx, 
  const bool black_border=false) {

  return edt::edt(labels, sx, wx, black_border);
}

template <typename T>
float* binary_edt(
    T* labels, 
    const int sx, const int sy, 
    const float wx, const float wy, 
    const bool black_border=false, const int parallel=1,
    float* output=NULL
  ) {

  return pyedt::_binary_edt2d(
    labels, 
    sx, sy, 
    wx, wy, 
    black_border, parallel, 
    output
  );
}

template <typename T>
float* binary_edt(
  T* labels, 
  const int sx, const int sy, const int sz, 
  const float wx, const float wy, const float wz,
  const bool black_border=false, const int parallel=1, float* output=NULL) {

  return pyedt::_binary_edt3d(labels, sx, sy, sz, wx, wy, wz, black_border, parallel, output);
}

template <typename T>
float* edtsq(
  T* labels, 
  const int sx, const float wx, 
  const bool black_border=false) {

  float* d = new float[sx]();
  pyedt::squared_edt_1d_multi_seg(labels, d, sx, 1, wx, black_border);
  return d;
}

template <typename T>
float* edtsq(
    T* labels, 
    const int sx, const int sy, 
    const float wx, const float wy,
    const bool black_border=false, const int parallel=1,
    float* output=NULL
  ) {

  return pyedt::_edt2dsq(labels, sx, sy, wx, wy, black_border, parallel, output);
}

template <typename T>
float* edtsq(
    T* labels, 
    const int sx, const int sy, const int sz, 
    const float wx, const float wy, const float wz,
    const bool black_border=false, const int parallel=1, 
    float* output=NULL
  ) {

  return pyedt::_edt3dsq(
    labels, 
    sx, sy, sz, 
    wx, wy, wz, 
    black_border, parallel, output
  );
}

template <typename T>
float* binary_edtsq(
  T* labels, 
  const int sx, const float wx, 
  const bool black_border=false, const int parallel=1) {

  return edt::edtsq(labels, sx, wx, black_border);
}

template <typename T>
float* binary_edtsq(
  T* labels, 
  const int sx, const int sy, 
  const float wx, const float wy,
  const bool black_border=false, const int parallel=1) {

  return pyedt::_binary_edt2dsq(labels, sx, sy, wx, wy, black_border, parallel);
}

template <typename T>
float* binary_edtsq(
  T* labels, 
  const int sx, const int sy, const int sz, 
  const float wx, const float wy, const float wz,
  const bool black_border=false, const int parallel=1, float* output=NULL) {

  return pyedt::_binary_edt3dsq(labels, sx, sy, sz, wx, wy, wz, parallel, output);
}


} // namespace edt

#undef sq

#endif


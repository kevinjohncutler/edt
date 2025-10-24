#pragma once

// Included from edt.hpp while inside namespace pyedt.

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

  for (z = 0; z < sz; z++) {
    pool.enqueue([workspace, sx, sy, sxy, z, wy, black_border](){
      for (size_t x = 0; x < sx; x++) {
        squared_edt_1d_parabolic_multi_seg<float>(
          (workspace + x + sxy * z),
          (workspace + x + sxy * z),
          sy, sx, wy, black_border
        );
      }
    });
  }

  pool.join();
  pool.start(parallel);

  for (x = 0; x < sx; x++) {
    pool.enqueue([workspace, sx, sy, sz, sxy, x, wz, black_border](){
      size_t y = 0;
      for (y = 0; y < sy; y++) {
        size_t loc = x + y * sx;
        if (workspace[loc]) {
          break;
        }
      }

      for (size_t z = 0; z < sz; z++) {
        squared_edt_1d_parabolic_multi_seg<float>(
          (workspace + x + y * sx + z * sxy), 
          (workspace + x + y * sx + z * sxy), 
          sz - z, sxy, wz,
          (black_border || (z > 0))
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

// Unified 2D squared EDT and features (renamed to _edt2dsq_with_features)
template <typename T, typename OUTIDX=size_t>
float* _edt2dsq_with_features(
    T* input,
    const size_t sx, const size_t sy,
    const float wx, const float wy,
    const bool black_border=false, const int parallel=1,
    float* output_dt=nullptr,
    OUTIDX* output_feat=nullptr
  ) {
  // If output_feat is nullptr, do regular EDT (three passes)
  const size_t voxels = sx * sy;
  if (output_feat == nullptr) {
    // Default: output_dt must not be nullptr
    float* workspace = output_dt;
    bool free_workspace = false;
    if (workspace == nullptr) {
      workspace = new float[voxels]();
      free_workspace = true;
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
    {
      ThreadPool pool(std::max(1, parallel));
      const size_t bx = std::max<size_t>(1, sx / (size_t(4) * std::max(1, parallel)));
      for (size_t x0 = 0; x0 < sx; x0 += bx) {
        const size_t x1 = std::min(sx, x0 + bx);
        pool.enqueue([=]() {
          std::vector<float> column(sy);
          std::vector<float> dcolumn(sy);
          for (size_t x = x0; x < x1; ++x) {
            for (size_t y = 0; y < sy; ++y) {
              column[y] = workspace[y * sx + x];
            }
            squared_edt_1d_parabolic_multi_seg<float>(
              column.data(), dcolumn.data(), sy, 1, wy, black_border
            );
            for (size_t y = 0; y < sy; ++y) {
              workspace[y * sx + x] = dcolumn[y];
            }
          }
        });
      }
      pool.join();
    }
    if (!black_border) {
      toinfinite(workspace, voxels);
    }
    if (free_workspace) {
      return workspace;
    }
    return output_dt;
  } else {
    ThreadPool pool(std::max(1, parallel));
    const size_t bx = std::max<size_t>(1, sx / (size_t(4) * std::max(1, parallel)));
    for (size_t x0 = 0; x0 < sx; x0 += bx) {
      const size_t x1 = std::min(sx, x0 + bx);
      pool.enqueue([=]() {
        std::vector<float> row(sx);
        std::vector<float> drow(sx);
        std::vector<int>   arow(sx);
        for (size_t y = 0; y < sy; ++y) {
          const size_t base = y * sx;
          for (size_t x = 0; x < sx; ++x) {
            row[x] = (input[base + x] != 0) ? 0.f : std::numeric_limits<float>::max() / 4.0f;
          }
          squared_edt_1d_parabolic_multi_seg<float>(
            row.data(), drow.data(), sx, 1, wx, false, arow.data()
          );
          for (size_t x = 0; x < sx; ++x) {
            output_dt[base + x] = drow[x];
            output_feat[base + x] = (OUTIDX)((arow[x] >= 0)
                                      ? (size_t(arow[x]) + size_t(y) * sx)
                                      : size_t(0));
          }
        }
      });
    }
    pool.join();
    {
      ThreadPool pool(std::max(1, parallel));
      const size_t by = std::max<size_t>(1, sy / (size_t(4) * std::max(1, parallel)));
      for (size_t y0 = 0; y0 < sy; y0 += by) {
        const size_t y1 = std::min(sy, y0 + by);
        pool.enqueue([=]() {
          std::vector<float> col(sy); std::vector<float> dcol(sy);
          std::vector<int> arow(sy);
          for (size_t x = 0; x < sx; ++x) {
            for (size_t y = y0; y < y1; ++y) {
              const size_t idx = y * sx + x;
              col[y - y0] = output_dt[idx];
            }
            squared_edt_1d_parabolic_multi_seg<float>(
              col.data(), dcol.data(), y1 - y0, 1, wy, false, arow.data()
            );
            for (size_t y = y0; y < y1; ++y) {
              const size_t idx = y * sx + x;
              output_dt[idx] = dcol[y - y0];
            }
          }
        });
      }
      pool.join();
    }
    return output_dt;
  }
}

// Unified 3D squared EDT and features (renamed to _edt3dsq_with_features)
template <typename T, typename OUTIDX=size_t>
float* _edt3dsq_with_features(
    T* input,
    const size_t sx, const size_t sy, const size_t sz,
    const float wx, const float wy, const float wz,
    const bool black_border=false, const int parallel=1,
    float* output_dt=nullptr,
    OUTIDX* output_feat=nullptr
  ) {
  const size_t sxy = sx * sy;
  const size_t voxels = sz * sxy;
  if (output_feat == nullptr) {
    float* workspace = output_dt;
    bool free_workspace = false;
    if (workspace == nullptr) {
      workspace = new float[voxels]();
      free_workspace = true;
    }
    ThreadPool pool(parallel);
    for (size_t z = 0; z < sz; z++) {
      pool.enqueue([input, sy, z, sx, sxy, wx, workspace, black_border]() {
        for (size_t y = 0; y < sy; y++) {
          squared_edt_1d_multi_seg<T>(
            (input + sx * y + sxy * z),
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
      pool.enqueue([input, sxy, z, workspace, sx, sy, wy, black_border]() {
        for (size_t x = 0; x < sx; x++) {
          squared_edt_1d_parabolic_multi_seg<T>(
            (input + x + sxy * z),
            (workspace + x + sxy * z),
            sy, sx, wy, black_border
          );
        }
      });
    }
    pool.join();
    pool.start(parallel);
    for (size_t y = 0; y < sy; y++) {
      pool.enqueue([input, sx, y, workspace, sz, sxy, wz, black_border]() {
        for (size_t x = 0; x < sx; x++) {
          squared_edt_1d_parabolic_multi_seg<T>(
            (input + x + sx * y),
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
    if (free_workspace) {
      return workspace;
    }
    return output_dt;
  } else {
    std::unique_ptr<float[]> dx(new float[voxels]);
    std::unique_ptr<int[]>   sx_seed(new int[voxels]);
    float* dx_p = dx.get();
    int*   sx_seed_p = sx_seed.get();
    {
      ThreadPool pool(std::max(1, parallel));
      const size_t by = std::max<size_t>(1, sy / (size_t(4) * std::max(1, parallel)));
      for (size_t z = 0; z < sz; ++z) {
        for (size_t y0 = 0; y0 < sy; y0 += by) {
          const size_t y1 = std::min(sy, y0 + by);
          pool.enqueue([=]() {
            std::vector<float> row(sx);
            std::vector<float> drow(sx);
            std::vector<int>   arow(sx);
            const size_t base_z = z * sxy;
            for (size_t y = y0; y < y1; ++y) {
              const size_t base = base_z + y * sx;
              for (size_t x = 0; x < sx; ++x) {
                row[x] = (input[base + x] != 0) ? 0.f : std::numeric_limits<float>::max() / 4.0f;
              }
              squared_edt_1d_parabolic_multi_seg<float>(
                row.data(), drow.data(), sx, 1, wx, false, arow.data()
              );
              for (size_t x = 0; x < sx; ++x) {
                dx_p[base + x] = drow[x];
                sx_seed_p[base + x] = arow[x];
              }
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
    {
      ThreadPool pool(std::max(1, parallel));
      const size_t bx = std::max<size_t>(1, sx / (size_t(4) * std::max(1, parallel)));
      for (size_t z = 0; z < sz; ++z) {
        for (size_t x0 = 0; x0 < sx; x0 += bx) {
          const size_t x1 = std::min(sx, x0 + bx);
          pool.enqueue([=]() {
            std::vector<float> dcol(sy); std::vector<int> arow(sy); std::vector<float> g(sy);
            std::vector<int> axcol(sy);
            for (size_t x = x0; x < x1; ++x) {
              for (size_t y = 0; y < sy; ++y) {
                const size_t idx = z * sxy + y * sx + x;
                g[y] = dx_p[idx];
                axcol[y] = sx_seed_p[idx];
              }
              squared_edt_1d_parabolic_multi_seg<float>(
                g.data(), dcol.data(), sy, 1, wy, false, arow.data()
              );
              for (size_t y = 0; y < sy; ++y) {
                const size_t idx = z * sxy + y * sx + x;
                dxy_p[idx] = dcol[y];
                const int r = arow[y];
                sy_seed_p[idx] = r;
                sx_seed_y_p[idx] = (r >= 0 ? axcol[r] : -1);
              }
            }
          });
        }
      }
      pool.join();
    }
    {
      ThreadPool pool(std::max(1, parallel));
      const size_t cols = sx * sy;
      const size_t bxy = std::max<size_t>(1, cols / (size_t(4) * std::max(1, parallel)));
      for (size_t c0 = 0; c0 < cols; c0 += bxy) {
        const size_t c1 = std::min(cols, c0 + bxy);
        pool.enqueue([=]() {
          std::vector<float> dline(sz); std::vector<int> arow(sz); std::vector<float> g(sz);
          std::vector<int> ayline(sz);
          for (size_t c = c0; c < c1; ++c) {
            const size_t y = c / sx; const size_t x = c % sx;
            for (size_t z = 0; z < sz; ++z) {
              const size_t idx = z * sxy + y * sx + x;
              g[z] = dxy_p[idx];
              ayline[z] = sy_seed_p[idx];
            }
            squared_edt_1d_parabolic_multi_seg<float>(
              g.data(), dline.data(), sz, 1, wz, false, arow.data()
            );
            for (size_t z = 0; z < sz; ++z) {
              const size_t idx = z * sxy + y * sx + x;
              if (output_dt) output_dt[idx] = dline[z];
              const int rz = arow[z];
              if (rz >= 0) {
                const int ry = ayline[rz];
                const int rx = (ry >= 0) ? sx_seed_y_p[rz * sxy + size_t(ry) * sx + x] : -1;
                output_feat[idx] = (OUTIDX)((rx >= 0 && ry >= 0)
                                     ? (size_t(rx) + sx * (size_t(ry) + sy * size_t(rz)))
                                     : size_t(0));
              } else {
                output_feat[idx] = (OUTIDX)0;
              }
            }
          }
        });
      }
      pool.join();
    }
    return output_dt;
  }
}

template <typename T>
float* _edt2dsq_features(
    T* labels, size_t sx, size_t sy,
    float wx, float wy,
    bool black_border, int parallel,
    float* output_dt, size_t* output_feat) {
  return _edt2dsq_with_features<T, size_t>(
      labels, sx, sy, wx, wy, black_border, parallel, output_dt, output_feat);
}

template <typename T>
float* _edt3dsq_features(
    T* labels, size_t sx, size_t sy, size_t sz,
    float wx, float wy, float wz,
    bool black_border, int parallel,
    float* output_dt, size_t* output_feat) {
  return _edt3dsq_with_features<T, size_t>(
      labels, sx, sy, sz, wx, wy, wz, black_border, parallel, output_dt, output_feat);
}

template <typename T>
float* _edt2dsq_features_u32(
    T* labels, size_t sx, size_t sy,
    float wx, float wy,
    bool black_border, int parallel,
    float* output_dt, uint32_t* output_feat) {
  return _edt2dsq_with_features<T, uint32_t>(
      labels, sx, sy, wx, wy, black_border, parallel, output_dt, output_feat);
}

template <typename T>
float* _edt3dsq_features_u32(
    T* labels, size_t sx, size_t sy, size_t sz,
    float wx, float wy, float wz,
    bool black_border, int parallel,
    float* output_dt, uint32_t* output_feat) {
  return _edt3dsq_with_features<T, uint32_t>(
      labels, sx, sy, sz, wx, wy, wz, black_border, parallel, output_dt, output_feat);
}

template <typename T>
void _expand2d_u32(
    T* labels, size_t sx, size_t sy,
    float wx, float wy,
    bool black_border, int parallel,
    uint32_t* out,
    const uint32_t* label_values
) {
    _edt2dsq_with_features<T,uint32_t>(
        labels,
        sx, sy,
        wx, wy,
        black_border, parallel,
        nullptr, out
    );
}

template <typename T>
void _expand3d_u32(
    T* labels, size_t sx, size_t sy, size_t sz,
    float wx, float wy, float wz,
    bool black_border, int parallel,
    uint32_t* out,
    const uint32_t* label_values
) {
    _edt3dsq_with_features<T,uint32_t>(
        labels,
        sx, sy, sz,
        wx, wy, wz,
        black_border, parallel,
        nullptr, out
    );
}

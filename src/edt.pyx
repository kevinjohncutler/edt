# cython: language_level=3
"""
Cython binding for the C++ multi-label Euclidean Distance
Transform library by William Silversmith based on the 
algorithms of Meijister et al (2002) Felzenzwalb et al. (2012) 
and Saito et al. (1994).

Given a 1d, 2d, or 3d volume of labels, compute the Euclidean
Distance Transform such that label boundaries are marked as
distance 1 and 0 is always 0.

Key methods: 
  edt, edtsq
  edt1d,   edt2d,   edt3d,
  edt1dsq, edt2dsq, edt3dsq

License: GNU 3.0

Author: William Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institute
Date: July 2018 - December 2023
"""
import operator
from functools import reduce
from libc.stdint cimport (
  uint8_t, uint16_t, uint32_t, uint64_t,
   int8_t,  int16_t,  int32_t,  int64_t
)
from libcpp cimport bool as native_bool
from libcpp.map cimport map as mapcpp
from libcpp.utility cimport pair as cpp_pair
from libcpp.vector cimport vector

import multiprocessing

import cython
from cython cimport floating
from cpython cimport array 
cimport numpy as np
np.import_array()

import numpy as np

ctypedef fused UINT:
  uint8_t
  uint16_t
  uint32_t
  uint64_t

ctypedef fused INT:
  int8_t
  int16_t
  int32_t
  int64_t

ctypedef fused NUMBER:
  UINT
  INT
  float
  double

from libc.stddef cimport size_t
from libc.stdlib cimport malloc, free
from libc.math cimport isinf, INFINITY
#
# Remove duplicate and update _edt2dsq/_edt3dsq for optional features
cdef extern from "edt.hpp" namespace "pyedt":
    cdef void _expand2d_u32[T](
        T* labels, size_t sx, size_t sy,
        float wx, float wy,
        native_bool black_border, int parallel,
        uint32_t* out,
        const uint32_t* label_values
    ) nogil

    cdef void _expand3d_u32[T](
        T* labels, size_t sx, size_t sy, size_t sz,
        float wx, float wy, float wz,
        native_bool black_border, int parallel,
        uint32_t* out,
        const uint32_t* label_values
    ) nogil

    cdef void squared_edt_1d_multi_seg[T](
        T *labels,
        float *dest,
        int n,
        int stride,
        float anisotropy,
        native_bool black_border
        ) nogil

    cdef void squared_edt_1d_parabolic_multi_seg[T](
        T *labels,
        float *dest,
        int n,
        int stride,
        float anisotropy,
        native_bool black_border
        ) nogil

    cdef void squared_edt_1d_parabolic_with_arg(
        float *f,
        int n,
        int stride,
        float anisotropy,
        native_bool black_border_left,
        native_bool black_border_right,
        int* arg_out
        ) nogil

    # Expose variant that returns argmin indices along the processed axis
    cdef void squared_edt_1d_parabolic_multi_seg_with_arg[T](
        T *labels,
        float *dest,
        int n,
        int stride,
        float anisotropy,
        native_bool black_border,
        int* arg_out
        ) nogil

    # ND expand helpers (threaded per-axis over precomputed bases)
    cdef void _nd_expand_init_bases[INDEX](
        uint8_t* seeds,
        float* dist,
        size_t* bases,
        size_t num_lines,
        size_t n,
        size_t s,
        float anis,
        native_bool black_border,
        INDEX* feat_out,
        int parallel
    ) nogil

    cdef void _nd_expand_parabolic_bases[INDEX](
        float* dist,
        size_t* bases,
        size_t num_lines,
        size_t n,
        size_t s,
        float anis,
        native_bool black_border,
        INDEX* feat_in,
        INDEX* feat_out,
        int parallel
    ) nogil

    cdef void _nd_expand_init_labels_bases(
        uint8_t* seeds,
        float* dist,
        size_t* bases,
        size_t num_lines,
        size_t n,
        size_t s,
        float anis,
        native_bool black_border,
        uint32_t* labelsp,
        uint32_t* label_out,
        int parallel
    ) nogil

    cdef void _nd_expand_parabolic_labels_bases(
        float* dist,
        size_t* bases,
        size_t num_lines,
        size_t n,
        size_t s,
        float anis,
        native_bool black_border,
        uint32_t* label_in,
        uint32_t* label_out,
        int parallel
    ) nogil

    # Updated _edt2dsq and _edt3dsq to allow optional feature tracking
    cdef float* _edt2dsq[T, OUTIDX](
        T* labels,
        size_t sx, size_t sy,
        float wx, float wy,
        native_bool black_border, int parallel,
        float* output_dt, OUTIDX* output_feat
    ) nogil

    cdef float* _edt3dsq[T, OUTIDX](
        T* labels,
        size_t sx, size_t sy, size_t sz,
        float wx, float wy, float wz,
        native_bool black_border, int parallel,
        float* output_dt, OUTIDX* output_feat
    ) nogil

    # Feature transform wrappers
    cdef float* _edt2dsq_features[T](
        T* labels, size_t sx, size_t sy,
        float wx, float wy,
        native_bool black_border, int parallel,
        float* output_dt, size_t* output_feat
    ) nogil

    cdef float* _edt3dsq_features[T](
        T* labels, size_t sx, size_t sy, size_t sz,
        float wx, float wy, float wz,
        native_bool black_border, int parallel,
        float* output_dt, size_t* output_feat
    ) nogil

    cdef float* _edt2dsq_features_u32[T](
        T* labels, size_t sx, size_t sy,
        float wx, float wy,
        native_bool black_border, int parallel,
        float* output_dt, uint32_t* output_feat
    ) nogil

    cdef float* _edt3dsq_features_u32[T](
        T* labels, size_t sx, size_t sy, size_t sz,
        float wx, float wy, float wz,
        native_bool black_border, int parallel,
        float* output_dt, uint32_t* output_feat
    ) nogil

    # Unified EDT+features (internal helpers)
    cdef float* _edt2dsq_with_features[T, OUTIDX](
        T* input, size_t sx, size_t sy,
        float wx, float wy,
        native_bool black_border, int parallel,
        float* output_dt, OUTIDX* output_feat
    ) nogil

    # Odometer threaded passes (no base arrays)
    cdef void _nd_pass_multi_odometer[T](
        T* labels, float* dest,
        size_t dims, size_t* shape, size_t* strides,
        size_t ax, float anis, native_bool black_border, int parallel
    ) nogil
    cdef void _nd_pass_parabolic_odometer[T](
        T* labels, float* dest,
        size_t dims, size_t* shape, size_t* strides,
        size_t ax, float anis, native_bool black_border, int parallel
    ) nogil

    cdef float* _edt3dsq_with_features[T, OUTIDX](
        T* input, size_t sx, size_t sy, size_t sz,
        float wx, float wy, float wz,
        native_bool black_border, int parallel,
        float* output_dt, OUTIDX* output_feat
    ) nogil

    # ND threaded axis helpers
    cdef void _nd_pass_multi[T](
        T* labels, float* dest,
        size_t dims, size_t* shape, size_t* strides,
        size_t ax, float anis, native_bool black_border, int parallel
    ) nogil
    cdef void _nd_pass_parabolic[T](
        T* labels, float* dest,
        size_t dims, size_t* shape, size_t* strides,
        size_t ax, float anis, native_bool black_border, int parallel
    ) nogil

    cdef void _nd_pass_multi_bases[T](
        T* labels, float* dest,
        size_t* bases, size_t num_lines,
        size_t n, size_t s,
        float anis, native_bool black_border, int parallel
    ) nogil
    cdef void _nd_pass_parabolic_bases[T](
        T* labels, float* dest,
        size_t* bases, size_t num_lines,
        size_t n, size_t s,
        float anis, native_bool black_border, int parallel
    ) nogil


cdef extern from "edt_voxel_graph.hpp" namespace "pyedt":
  cdef float* _edt2dsq_voxel_graph[T,GRAPH_TYPE](
    T* labels, GRAPH_TYPE* graph,
    size_t sx, size_t sy,
    float wx, float wy,
    native_bool black_border, float* workspace
  ) nogil 
  cdef float* _edt3dsq_voxel_graph[T,GRAPH_TYPE](
    T* labels, GRAPH_TYPE* graph,
    size_t sx, size_t sy, size_t sz, 
    float wx, float wy, float wz,
    native_bool black_border, float* workspace
  ) nogil
  cdef mapcpp[T, vector[cpp_pair[size_t,size_t]]] extract_runs[T](
    T* labels, size_t voxels
  )
  void set_run_voxels[T](
    T key,
    vector[cpp_pair[size_t, size_t]] all_runs,
    T* labels, size_t voxels
  ) except +
  void transfer_run_voxels[T](
    vector[cpp_pair[size_t, size_t]] all_runs,
    T* src, T* dest,
    size_t voxels
  ) except +

def nvl(val, default_val):
  if val is None:
    return default_val
  return val

@cython.binding(True)
def sdf(
  data, anisotropy=None, black_border=False,
  int parallel = 1, voxel_graph=None, order=None
):
  """
  Computes the anisotropic Signed Distance Function (SDF) using the Euclidean
  Distance Transform (EDT) of up to 3D numpy arrays. The SDF is the same as the
  EDT except that the background (zero) color is also processed and assigned a 
  negative distance.

  Supported Data Types:
    (u)int8, (u)int16, (u)int32, (u)int64, 
     float32, float64, and boolean

  Required:
    data: a 1d, 2d, or 3d numpy array with a supported data type.
  Optional:
    anisotropy:
      1D: scalar (default: 1.0)
      2D: (x, y) (default: (1.0, 1.0) )
      3D: (x, y, z) (default: (1.0, 1.0, 1.0) )
    black_border: (boolean) if true, consider the edge of the
      image to be surrounded by zeros.
    parallel: number of threads to use (only applies to 2D and 3D)
    order: no longer functional, for backwards compatibility
  Returns: SDF of data
  """
  def fn(labels):
    return edt(
      labels,
      anisotropy=anisotropy,
      black_border=black_border,
      parallel=parallel,
      voxel_graph=voxel_graph,
    )
  return fn(data) - fn(data == 0)

@cython.binding(True)
def sdfsq(
  data, anisotropy=None, black_border=False,
  int parallel = 1, voxel_graph=None
):
  """
  sdfsq(data, anisotropy=None, black_border=False, order="K", parallel=1)

  Computes the squared anisotropic Signed Distance Function (SDF) using the Euclidean
  Distance Transform (EDT) of up to 3D numpy arrays. The SDF is the same as the
  EDT except that the background (zero) color is also processed and assigned a 
  negative distance.

  data is assumed to be memory contiguous in either C (XYZ) or Fortran (ZYX) order. 
  The algorithm works both ways, however you'll want to reverse the order of the
  anisotropic arguments for Fortran order.

  Supported Data Types:
    (u)int8, (u)int16, (u)int32, (u)int64, 
     float32, float64, and boolean

  Required:
    data: a 1d, 2d, or 3d numpy array with a supported data type.
  Optional:
    anisotropy:
      1D: scalar (default: 1.0)
      2D: (x, y) (default: (1.0, 1.0) )
      3D: (x, y, z) (default: (1.0, 1.0, 1.0) )
    black_border: (boolean) if true, consider the edge of the
      image to be surrounded by zeros.
    parallel: number of threads to use (only applies to 2D and 3D)

  Returns: squared SDF of data
  """
  def fn(labels):
    return edtsq(
      labels,
      anisotropy=anisotropy,
      black_border=black_border,
      parallel=parallel,
      voxel_graph=voxel_graph,
    )
  return fn(data) - fn(data == 0)

@cython.binding(True)
def edt(
    data, anisotropy=None, black_border=False, 
    int parallel=1, voxel_graph=None, order=None,
  ):
  """
  Computes the anisotropic Euclidean Distance Transform (EDT) of 1D, 2D, or 3D numpy arrays.

  data is assumed to be memory contiguous in either C (XYZ) or Fortran (ZYX) order. 
  The algorithm works both ways, however you'll want to reverse the order of the
  anisotropic arguments for Fortran order.

  Supported Data Types:
    (u)int8, (u)int16, (u)int32, (u)int64, 
     float32, float64, and boolean

  Required:
    data: a 1d, 2d, or 3d numpy array with a supported data type.
  Optional:
    anisotropy:
      1D: scalar (default: 1.0)
      2D: (x, y) (default: (1.0, 1.0) )
      3D: (x, y, z) (default: (1.0, 1.0, 1.0) )
    black_border: (boolean) if true, consider the edge of the
      image to be surrounded by zeros.
    parallel: number of threads to use (only applies to 2D and 3D)
    voxel_graph: A numpy array where each voxel contains a  bitfield that 
      represents a directed graph of the allowed directions for transit 
      between voxels. If a connection is allowed, the respective direction 
      is set to 1 else it set to 0.

      See https://github.com/seung-lab/connected-components-3d/blob/master/cc3d.pyx#L743-L783
      for details.
    order: no longer functional, for backwards compatibility

  Returns: EDT of data
  """
  dt = edtsq(data, anisotropy, black_border, parallel, voxel_graph)
  return np.sqrt(dt,dt)

@cython.binding(True)
def edtsq(
  data, anisotropy=None, native_bool black_border=False, 
  int parallel=1, voxel_graph=None, order=None,
):
  """
  Computes the squared anisotropic Euclidean Distance Transform (EDT) of 1D, 2D, or 3D numpy arrays.

  Squaring allows for omitting an sqrt operation, so may be faster if your use case allows for it.

  data is assumed to be memory contiguous in either C (XYZ) or Fortran (ZYX) order. 
  The algorithm works both ways, however you'll want to reverse the order of the
  anisotropic arguments for Fortran order.

  Supported Data Types:
    (u)int8, (u)int16, (u)int32, (u)int64, 
     float32, float64, and boolean

  Required:
    data: a 1d, 2d, or 3d numpy array with a supported data type.
  Optional:
    anisotropy:
      1D: scalar (default: 1.0)
      2D: (x, y) (default: (1.0, 1.0) )
      3D: (x, y, z) (default: (1.0, 1.0, 1.0) )
    black_border: (boolean) if true, consider the edge of the
      image to be surrounded by zeros.
    parallel: number of threads to use (only applies to 2D and 3D)
    order: no longer functional, for backwards compatibility

  Returns: Squared EDT of data
  """
  if isinstance(data, list):
    data = np.array(data)

  dims = len(data.shape)

  if data.size == 0:
    return np.zeros(shape=data.shape, dtype=np.float32)

  order = 'F' if data.flags.f_contiguous else 'C'
  if not data.flags.c_contiguous and not data.flags.f_contiguous:
    data = np.ascontiguousarray(data)

  if parallel <= 0:
    parallel = multiprocessing.cpu_count()
  else:
    # Cap to physical cores to avoid oversubscription with MKL/BLAS
    try:
      parallel = max(1, min(parallel, multiprocessing.cpu_count()))
    except Exception:
      parallel = max(1, parallel)

  if voxel_graph is not None and dims not in (2,3):
    raise TypeError("Voxel connectivity graph is only supported for 2D and 3D. Got {}.".format(dims))

  if voxel_graph is not None:
    if order == 'C':
      voxel_graph = np.ascontiguousarray(voxel_graph)
    else:
      voxel_graph = np.asfortranarray(voxel_graph)

  if dims == 1:
    anisotropy = nvl(anisotropy, 1.0)
    return edt1dsq(data, anisotropy, black_border)
  elif dims == 2:
    anisotropy = nvl(anisotropy, (1.0, 1.0))
    return edt2dsq(data, anisotropy, black_border, parallel=parallel, voxel_graph=voxel_graph)
  elif dims == 3:
    anisotropy = nvl(anisotropy, (1.0, 1.0, 1.0))
    return edt3dsq(data, anisotropy, black_border, parallel=parallel, voxel_graph=voxel_graph)
  else:
    raise TypeError("Multi-Label EDT library only supports up to 3 dimensions got {}.".format(dims))

@cython.binding(True)
def feature_transform(data, anisotropy=None, black_border=False,
                      int parallel=1, voxel_graph=None,
                      return_distances=False, features_dtype='auto'):
  """
  Feature transform of a label/seed image.

  Nonzero elements are seeds. Returns an array of dtype np.intp (or np.uintp)
  with the linear index of the nearest seed for every voxel. If
  return_distances=True, also returns the squared EDT.

  Parameters
  ----------
  data : ndarray
      The input array (nonzero elements are seeds).
  anisotropy : tuple, optional
      Voxel size per axis (default 1.0 for each axis).
  black_border : bool, optional
      If True, treat the border as background (default False).
  parallel : int, optional
      Number of threads for parallel execution.
  voxel_graph : ndarray, optional
      Not used.
  return_distances : bool, optional
      If True, also return the squared EDT.
  ties : {'last', 'first', 'scipy', 'left', 'smallest'}, optional
      Tie-breaking policy for voxels equidistant to multiple seeds.
      'last' (default): prefers the last site encountered (largest index, matches NumPy).
      'first', 'scipy', 'left', 'smallest': prefers the first (lexicographically smallest index, matches SciPy).
      All aliases besides 'last' map to the same behavior ('first').
  features_dtype : {'auto', 'uint32', 'u32', 'uintp'}, optional
      Output dtype for the feature (nearest seed) array. 'auto' (default) uses uint32 if the
      number of voxels fits; otherwise uses uintp. 'uint32' or 'u32' always uses np.uint32,
      'uintp' always uses np.uintp.
  Returns
  -------
  feat : ndarray of intp or uint32
      Linear indices of the nearest seed for each voxel.
  dt : ndarray of float32, optional
      Squared Euclidean distance, if return_distances=True.
  """
  # --- Cython declarations at function top-level ---
  cdef native_bool bb
  cdef int dims
  cdef tuple anis
  cdef np.ndarray arr

  # 1D
  cdef np.ndarray[np.uint8_t, ndim=1] seeds1
  cdef Py_ssize_t n1 = 0
  cdef np.ndarray[np.float32_t, ndim=1] dt1
  cdef np.ndarray feat1
  cdef float wx1
  cdef Py_ssize_t i1 = 0
  cdef Py_ssize_t s1 = 0
  cdef Py_ssize_t k1 = 0

  # 2D
  cdef np.ndarray[np.uint8_t,  ndim=2] seeds2
  cdef Py_ssize_t sx2 = 0
  cdef Py_ssize_t sy2 = 0
  cdef np.ndarray[np.float32_t, ndim=2] dt2
  cdef np.ndarray[np.uint32_t, ndim=2] feat2_u32
  cdef np.ndarray[np.uintp_t,  ndim=2] feat2_up
  cdef float* dt2_ptr = NULL

  # 3D
  cdef np.ndarray[np.uint8_t,  ndim=3] seeds3
  cdef Py_ssize_t sx3 = 0
  cdef Py_ssize_t sy3 = 0
  cdef Py_ssize_t sz3 = 0
  cdef np.ndarray[np.float32_t, ndim=3] dt3
  cdef np.ndarray[np.uint32_t, ndim=3] feat3_u32
  cdef np.ndarray[np.uintp_t,  ndim=3] feat3_up
  cdef float* dt3_ptr = NULL

  # --- Python-level setup after declarations ---
  bb = black_border
  arr = np.asarray(data)
  if arr.ndim not in (1, 2, 3):
    raise ValueError("Only 1D, 2D, 3D supported")
  if not arr.flags.c_contiguous:
    arr = np.ascontiguousarray(arr)

  dims = arr.ndim
  if anisotropy is None:
    anis = (1.0,) * dims
  else:
    anis = tuple(anisotropy) if hasattr(anisotropy, "__len__") else (float(anisotropy),) * dims
    if len(anis) != dims:
      raise ValueError("anisotropy length must match data.ndim")

  # Normalize parallel just like edtsq():
  if parallel <= 0:
    try:
      parallel = multiprocessing.cpu_count()
    except Exception:
      parallel = 1
  else:
    try:
      parallel = max(1, min(parallel, multiprocessing.cpu_count()))
    except Exception:
      parallel = max(1, parallel)

  # Decide features_dtype
  voxels = arr.size
  use_u32 = False
  if isinstance(features_dtype, str):
    fd = features_dtype.lower()
    if fd == 'auto':
      use_u32 = (voxels < 2**32)
    elif fd in ('uint32', 'u32'):
      use_u32 = True
    else:
      use_u32 = False  # 'uintp' or anything else

  # 1D path
  if dims == 1:
    seeds1 = np.where(arr != 0, 1, 0).astype(np.uint8, order='C', copy=False)
    n1 = seeds1.shape[0]
    dt1   = np.empty((n1,), dtype=np.float32)
    feat1 = np.empty((n1,), dtype=np.uint32 if use_u32 else np.uintp)
    wx1 = <float>anis[0]

    # compute nearest seed via simple midpoint selection
    pos = np.flatnonzero(seeds1)
    if pos.size == 0:
      for i1 in range(n1):
        feat1[i1] = i1
        dt1[i1] = np.inf
    else:
      mids = (pos[:-1] + pos[1:]) * 0.5 if pos.size > 1 else np.array([])
      k1 = 0
      for i1 in range(n1):
        while k1 < mids.size and i1 > mids[k1]:
          k1 += 1
        s1 = <Py_ssize_t>pos[min(k1, pos.size - 1)]
        feat1[i1] = s1
        dt1[i1] = wx1 * wx1 * (i1 - s1) * (i1 - s1)

    return (feat1, dt1) if return_distances else feat1

  # 2D path
  if dims == 2:
    seeds2 = np.where(arr != 0, 1, 0).astype(np.uint8, order='C', copy=False)
    sx2 = seeds2.shape[0]
    sy2 = seeds2.shape[1]
    if return_distances:
        dt2 = np.empty((sx2, sy2), dtype=np.float32)
        dt2_ptr = &dt2[0, 0]

    if use_u32:
        feat2_u32 = np.empty((sx2, sy2), dtype=np.uint32)
        _edt2dsq_features_u32[uint8_t](
            &seeds2[0, 0],
            <size_t>sx2, <size_t>sy2,
            <float>anis[0], <float>anis[1],
            bb, parallel,
            dt2_ptr, &feat2_u32[0, 0]
        )
        return (feat2_u32, dt2) if return_distances else feat2_u32
    else:
        feat2_up = np.empty((sx2, sy2), dtype=np.uintp)
        _edt2dsq_features[uint8_t](
            &seeds2[0, 0],
            <size_t>sx2, <size_t>sy2,
            <float>anis[0], <float>anis[1],
            bb, parallel,
            dt2_ptr, <size_t*>&feat2_up[0, 0]
        )
        return (feat2_up, dt2) if return_distances else feat2_up

  # 3D path
  seeds3 = np.where(arr != 0, 1, 0).astype(np.uint8, order='C', copy=False)
  sx3 = seeds3.shape[0]
  sy3 = seeds3.shape[1]
  sz3 = seeds3.shape[2]
  if return_distances:
      dt3 = np.empty((sx3, sy3, sz3), dtype=np.float32)
      dt3_ptr = &dt3[0, 0, 0]

  if use_u32:
      feat3_u32 = np.empty((sx3, sy3, sz3), dtype=np.uint32)
      _edt3dsq_features_u32[uint8_t](
          &seeds3[0, 0, 0],
          <size_t>sx3, <size_t>sy3, <size_t>sz3,
          <float>anis[0], <float>anis[1], <float>anis[2],
          bb, parallel,
          dt3_ptr, &feat3_u32[0, 0, 0]
      )
      return (feat3_u32, dt3) if return_distances else feat3_u32
  else:
      feat3_up = np.empty((sx3, sy3, sz3), dtype=np.uintp)
      _edt3dsq_features[uint8_t](
          &seeds3[0, 0, 0],
          <size_t>sx3, <size_t>sy3, <size_t>sz3,
          <float>anis[0], <float>anis[1], <float>anis[2],
          bb, parallel,
          dt3_ptr, <size_t*>&feat3_up[0, 0, 0]
      )
      return (feat3_up, dt3) if return_distances else feat3_up


# ================================
# Fast label expansion (no distances)
# ================================
@cython.binding(True)
def expand_labels(
    data, anisotropy=None, black_border=False,
    int parallel=1, voxel_graph=None):
  """Expand nonzero labels by nearest-neighbor in Euclidean metric (no distances).

  Returns an array of dtype uint32 with labels copied from nearest seed.
  """
  cdef native_bool bb = black_border
  # Enforce C-contiguous np.uint32 input
  cdef np.ndarray arr = np.require(data, dtype=np.uint32, requirements='C')
  if arr.ndim not in (1,2,3):
    raise ValueError('Only 1D, 2D, 3D supported')

  cdef int dims = arr.ndim
  cdef tuple anis
  if anisotropy is None:
    anis = (1.0,) * dims
  else:
    anis = tuple(anisotropy) if hasattr(anisotropy, '__len__') else (float(anisotropy),) * dims
    if len(anis) != dims:
      raise ValueError('anisotropy length must match data.ndim')

  if parallel <= 0:
    try: parallel = multiprocessing.cpu_count()
    except Exception: parallel = 1
  else:
    try: parallel = max(1, min(parallel, multiprocessing.cpu_count()))
    except Exception: parallel = max(1, parallel)

  # --- Cython variable declarations for all branches ---
  cdef Py_ssize_t n1
  cdef np.ndarray[np.uint32_t, ndim=1] out1
  cdef np.ndarray seed_pos
  cdef np.ndarray mids_arr
  cdef Py_ssize_t i
  cdef Py_ssize_t k
  cdef Py_ssize_t s

  cdef Py_ssize_t sx
  cdef Py_ssize_t sy
  cdef np.ndarray[np.uint32_t, ndim=2] out2
  cdef np.ndarray[np.uint8_t, ndim=2] seeds2
  cdef np.ndarray[np.uint32_t, ndim=2] arr2

  cdef Py_ssize_t sx3
  cdef Py_ssize_t sy3
  cdef Py_ssize_t sz3
  cdef np.ndarray[np.uint32_t, ndim=3] out3
  cdef np.ndarray[np.uint8_t, ndim=3] seeds3
  cdef np.ndarray[np.uint32_t, ndim=3] arr3
  cdef np.ndarray[np.uint32_t, ndim=1] label_values
  cdef np.ndarray[np.uint32_t, ndim=2] feat_idx2
  cdef np.ndarray[np.uint32_t, ndim=3] feat_idx3
  cdef uint32_t[:] label_values_view
  cdef uint8_t[:, :] seeds2_view
  cdef uint8_t[:, :, :] seeds3_view
  cdef uint32_t[:, :] feat_idx2_view
  cdef uint32_t[:, :, :] feat_idx3_view


  # 1D: simple midpoint selection
  if dims == 1:
    n1 = arr.shape[0]
    out1 = np.empty((n1,), dtype=np.uint32)
    # Detect seeds directly on arr
    pos = np.flatnonzero(arr)
    i = 0
    k = 0
    if pos.size == 0:
      out1.fill(0)
    elif pos.size == 1:
      out1.fill(<np.uint32_t>arr[int(pos[0])])
    else:
      mids = (pos[:-1] + pos[1:]) * 0.5
      for i in range(n1):
        while k < mids.size and i >= mids[k]:
          k += 1
        s = <Py_ssize_t>pos[min(k, pos.size-1)]
        out1[i] = <np.uint32_t>arr[s]
    return out1

  # 2D
  if dims == 2:
    # Optimized path using C++ feature transform kernels (like 3D)
    sx = arr.shape[1]
    sy = arr.shape[0]
    out2 = np.empty((sy, sx), dtype=np.uint32)
    seeds2 = (arr != 0).astype(np.uint8, order='C', copy=False)
    # Create label_values as a 1D array of the original array values (for indexing)
    label_values = arr.ravel().astype(np.uint32, order='C')
    feat_idx2 = np.empty((sy, sx), dtype=np.uint32)
    seeds2_view = seeds2
    feat_idx2_view = feat_idx2
    label_values_view = label_values
    _expand2d_u32[uint8_t](
        <uint8_t*>&seeds2_view[0, 0],
        <size_t>sx, <size_t>sy,
        <float>anis[1], <float>anis[0],
        bb, parallel,
        <uint32_t*>&feat_idx2_view[0, 0],
        <const uint32_t*>&label_values_view[0]
    )
    # Map feature indices to label values
    out2.ravel()[:] = label_values[feat_idx2.ravel()]
    return out2

  # 3D
  if dims == 3:
    # C-order 3D arrays are (z, y, x). Kernels expect sizes (sx=x, sy=y, sz=z).
    sx3 = arr.shape[2]
    sy3 = arr.shape[1]
    sz3 = arr.shape[0]
    out3 = np.empty((sz3, sy3, sx3), dtype=np.uint32)
    seeds3 = (arr != 0).astype(np.uint8, order='C', copy=False)
    # Create label_values as a 1D array of the original array values (for indexing)
    label_values = arr.ravel().astype(np.uint32, order='C')
    label_values_view = label_values
    # Allocate a temporary feature index buffer
    feat_idx3 = np.empty((sz3, sy3, sx3), dtype=np.uint32)
    seeds3_view = seeds3
    feat_idx3_view = feat_idx3
    # label_values_view is already set from 2D case above
    _expand3d_u32[uint8_t](
        <uint8_t*>&seeds3_view[0, 0, 0],
        <size_t>sx3, <size_t>sy3, <size_t>sz3,
        <float>anis[2], <float>anis[1], <float>anis[0],
        bb, parallel,
        <uint32_t*>&feat_idx3_view[0, 0, 0],
        <const uint32_t*>&label_values_view[0]
    )
    # Map feature indices to label values
    out3.ravel()[:] = label_values[feat_idx3.ravel()]
  return out3


@cython.binding(True)
def expand_labels_nd(
    data, anisotropy=None, black_border=False,
    int parallel=1, voxel_graph=None, return_features=False
  ):
  """Expand nonzero labels to zeros by nearest-neighbor in Euclidean metric (ND).

  This is a unified ND implementation (1D, 2D, 3D, …) that performs an
  in-place multi-axis lower-envelope transform with true argmin tracking.
  It early-exits trivial lines (all seeds on axis 0; all zeros on later axes)
  and, by default, propagates labels directly (no full feature map) for speed.

  Parameters
  ----------
  data : ndarray
      Input array; nonzero elements are seeds whose values are the labels.
      Any integer dtype is accepted; internally cast to uint32 for labels.
  anisotropy : float or sequence of float, optional
      Per-axis voxel size (default 1.0 for all axes).
  black_border : bool, optional
      If True, treat the border as background (default False).
  parallel : int, optional
      Number of threads; if <= 0, uses cpu_count().
  voxel_graph : ignored
      Present for API parity; not used in ND path.
  return_features : bool, optional
      If True, also return the feature (nearest-seed linear index) array.

  Returns
  -------
  labels : ndarray, dtype=uint32
      Expanded labels, same shape as input.
  features : ndarray, optional
      If return_features=True, the nearest-seed linear indices (uint32 or uintp)
      as a second return value.
  """
  cdef np.ndarray[np.uint32_t, ndim=1] out1
  cdef np.ndarray pos
  cdef np.ndarray mids
  cdef Py_ssize_t i = 0
  cdef Py_ssize_t k = 0
  cdef Py_ssize_t s
  # Declarations for optional feature-return path
  cdef np.ndarray feat_prev
  cdef np.ndarray feat_next
  cdef np.ndarray tmpF
  cdef uint32_t* fprev_u32
  cdef uint32_t* fnext_u32
  cdef size_t* fprev_sz
  cdef size_t* fnext_sz
  cdef np.ndarray[np.uint32_t, ndim=1] out_flat
  cdef np.uint32_t* outp
  cdef np.uint32_t* labp2
  cdef size_t idx2

  if parallel <= 0:
    try:
      parallel = multiprocessing.cpu_count()
    except Exception:
      parallel = 1
  else:
    try:
      parallel = max(1, min(parallel, multiprocessing.cpu_count()))
    except Exception:
      parallel = max(1, parallel)

  arr = np.require(data, dtype=np.uint32, requirements='C')
  dims = arr.ndim
  # Build anisotropy tuple
  cdef tuple anis
  if anisotropy is None:
    anis = (1.0,) * dims
  else:
    anis = tuple(anisotropy) if hasattr(anisotropy, '__len__') else (float(anisotropy),) * dims
    if len(anis) != dims:
      raise ValueError('anisotropy length must match data.ndim')
  if dims == 1:
    # reuse 1D midpoint selection from expand_labels 1D branch
    n1 = arr.shape[0]
    out1 = np.empty((n1,), dtype=np.uint32)
    seed_pos = np.flatnonzero(arr)
    if seed_pos.size == 0:
      out1.fill(0)
      return (out1, np.zeros((n1,), dtype=np.uintp)) if return_features else out1
    if seed_pos.size == 1:
      out1.fill(<np.uint32_t>arr[int(seed_pos[0])])
      if return_features:
        feat1 = np.full((n1,), int(seed_pos[0]), dtype=np.uintp)
        return out1, feat1
      return out1
    mids_arr = (seed_pos[:-1] + seed_pos[1:]) * 0.5
    for i in range(n1):
      while k < mids_arr.size and i >= mids_arr[k]:
        k += 1
      s = <Py_ssize_t>seed_pos[min(k, seed_pos.size-1)]
      out1[i] = <np.uint32_t>arr[s]
    if return_features:
      feat1 = np.empty((n1,), dtype=np.uintp)
      k = 0
      for i in range(n1):
        while k < mids_arr.size and i >= mids_arr[k]:
          k += 1
        feat1[i] = <Py_ssize_t>seed_pos[min(k, seed_pos.size-1)]
      return out1, feat1
    return out1
  elif dims == 2 or dims == 3:
    pass  # fall through to general ND implementation
  else:
    pass  # general ND implementation

  # General ND implementation (dims >= 2)
  cdef native_bool bb = black_border
  # Work with C-contiguous buffer
  if not arr.flags.c_contiguous:
    arr = np.ascontiguousarray(arr)
  cdef np.ndarray[np.uint8_t, ndim=1] seeds_flat = (arr.ravel(order='K') != 0).astype(np.uint8, order='C')
  cdef np.ndarray[np.uint32_t, ndim=1] labels_flat = arr.ravel(order='K').astype(np.uint32, order='C')
  cdef Py_ssize_t nd = arr.ndim
  cdef tuple shape = arr.shape
  # element strides for C-order
  cdef size_t* cshape = <size_t*> malloc(nd * sizeof(size_t))
  cdef size_t* cstrides = <size_t*> malloc(nd * sizeof(size_t))
  cdef Py_ssize_t* paxes = <Py_ssize_t*> malloc(nd * sizeof(Py_ssize_t))
  cdef float* canis = <float*> malloc(nd * sizeof(float))
  if cshape == NULL or cstrides == NULL or paxes == NULL or canis == NULL:
    if cshape != NULL: free(cshape)
    if cstrides != NULL: free(cstrides)
    if paxes != NULL: free(paxes)
    if canis != NULL: free(canis)
    raise MemoryError('Allocation failure in expand_labels_nd')
  cdef Py_ssize_t i_ax
  cdef Py_ssize_t ii
  for ii in range(nd):
    cshape[ii] = <size_t>shape[ii]
    canis[ii] = <float>anis[ii]
  # C-order strides
  cstrides[nd-1] = 1
  for ii in range(nd-2, -1, -1):
    cstrides[ii] = cstrides[ii+1] * cshape[ii+1]
  # Determine pass axis order by increasing stride, tie-break larger extent
  for ii in range(nd): paxes[ii] = ii
  cdef Py_ssize_t j
  cdef Py_ssize_t keyi
  for ii in range(1, nd):
    keyi = paxes[ii]
    j = ii - 1
    while j >= 0 and (cstrides[paxes[j]] > cstrides[keyi] or (cstrides[paxes[j]] == cstrides[keyi] and cshape[paxes[j]] < cshape[keyi])):
      paxes[j+1] = paxes[j]
      j -= 1
    paxes[j+1] = keyi

  # Allocate work buffers
  cdef size_t total = 1
  for ii in range(nd): total *= cshape[ii]
  cdef np.ndarray[np.float32_t, ndim=1] dist = np.empty((total,), dtype=np.float32)
  cdef float* distp = <float*> np.PyArray_DATA(dist)
  # Allocate either label propagation buffers or feature buffers
  cdef bint use_u32_feat = (total < (1<<32))
  cdef np.ndarray[np.uint32_t, ndim=1] lab_prev
  cdef np.ndarray[np.uint32_t, ndim=1] lab_next
  cdef np.ndarray[np.uint32_t, ndim=1] tmpL
  cdef uint32_t* labp = <uint32_t*> np.PyArray_DATA(labels_flat)
  cdef uint32_t* lprev
  cdef uint32_t* lnext
  feat_prev = None
  feat_next = None
  tmpF = None
  fprev_u32 = NULL
  fnext_u32 = NULL
  fprev_sz = NULL
  fnext_sz = NULL
  if return_features:
    if use_u32_feat:
      feat_prev = np.empty((total,), dtype=np.uint32)
      feat_next = np.empty((total,), dtype=np.uint32)
      fprev_u32 = <uint32_t*> np.PyArray_DATA(feat_prev)
      fnext_u32 = <uint32_t*> np.PyArray_DATA(feat_next)
    else:
      feat_prev = np.empty((total,), dtype=np.uintp)
      feat_next = np.empty((total,), dtype=np.uintp)
      fprev_sz = <size_t*> np.PyArray_DATA(feat_prev)
      fnext_sz = <size_t*> np.PyArray_DATA(feat_next)
  else:
    lab_prev = np.empty((total,), dtype=np.uint32)
    lab_next = np.empty((total,), dtype=np.uint32)
    lprev = <uint32_t*> np.PyArray_DATA(lab_prev)
    lnext = <uint32_t*> np.PyArray_DATA(lab_next)
  cdef uint8_t* seedsp = <uint8_t*> np.PyArray_DATA(seeds_flat)
  cdef uint32_t* labelsp = <uint32_t*> np.PyArray_DATA(labels_flat)

  # Axis 0 pass (threaded)
  cdef Py_ssize_t ax0 = paxes[0]
  cdef size_t n0 = cshape[ax0]
  cdef size_t s0 = cstrides[ax0]
  cdef size_t lines = total // n0
  # Precompute maximum lines across all passes to size bases safely
  cdef size_t max_lines = lines
  for ii in range(1, nd):
    if total // cshape[paxes[ii]] > max_lines:
      max_lines = total // cshape[paxes[ii]]
  cdef size_t* bases = <size_t*> malloc(max_lines * sizeof(size_t))
  if bases == NULL:
    free(cshape); free(cstrides); free(paxes); free(canis)
    raise MemoryError('Allocation failure for bases')
  # Build ord (other axes) by increasing stride
  cdef Py_ssize_t ord_len = nd - 1
  cdef Py_ssize_t* ord = <Py_ssize_t*> malloc(ord_len * sizeof(Py_ssize_t))
  if ord == NULL:
    free(bases); free(cshape); free(cstrides); free(paxes); free(canis)
    raise MemoryError('Allocation failure for ord')
  cdef Py_ssize_t ord_pos = 0
  for ii in range(nd):
    if ii != ax0:
      ord[ord_pos] = ii
      ord_pos += 1
  for i_ax in range(1, ord_len):
    keyi = ord[i_ax]
    j = i_ax - 1
    while j >= 0 and (cstrides[ord[j]] > cstrides[keyi] or (cstrides[ord[j]] == cstrides[keyi] and cshape[ord[j]] < cshape[keyi])):
      ord[j+1] = ord[j]
      j -= 1
    ord[j+1] = keyi
  cdef size_t il
  cdef size_t base_b
  cdef size_t tmp_b
  cdef size_t coord_b
  with nogil:
    for il in range(lines):
      base_b = 0
      tmp_b = il
      for j in range(ord_len):
        coord_b = tmp_b % cshape[ord[j]]
        base_b += coord_b * cstrides[ord[j]]
        tmp_b //= cshape[ord[j]]
      bases[il] = base_b
  if return_features:
    with nogil:
      if use_u32_feat:
        _nd_expand_init_bases[uint32_t](seedsp, distp, bases, lines, n0, s0, canis[ax0], bb, fprev_u32, parallel)
      else:
        _nd_expand_init_bases[size_t](seedsp, distp, bases, lines, n0, s0, canis[ax0], bb, fprev_sz, parallel)
  else:
    with nogil:
      _nd_expand_init_labels_bases(seedsp, distp, bases, lines, n0, s0, canis[ax0], bb, labp, lprev, parallel)

  # Remaining axes
  cdef Py_ssize_t a
  for a in range(1, nd):
    lines = total // cshape[paxes[a]]
    # rebuild ord for this axis
    free(ord)
    ord_len = nd - 1
    ord = <Py_ssize_t*> malloc(ord_len * sizeof(Py_ssize_t))
    if ord == NULL:
      free(bases); free(cshape); free(cstrides); free(paxes); free(canis)
      raise MemoryError('Allocation failure for ord')
    ord_pos = 0
    for ii in range(nd):
      if ii != paxes[a]:
        ord[ord_pos] = ii
        ord_pos += 1
    for i_ax in range(1, ord_len):
      keyi = ord[i_ax]
      j = i_ax - 1
      while j >= 0 and (cstrides[ord[j]] > cstrides[keyi] or (cstrides[ord[j]] == cstrides[keyi] and cshape[ord[j]] < cshape[keyi])):
        ord[j+1] = ord[j]
        j -= 1
      ord[j+1] = keyi
    # recompute bases for this axis
    with nogil:
      for il in range(lines):
        base_b = 0
        tmp_b = il
        for j in range(ord_len):
          coord_b = tmp_b % cshape[ord[j]]
          base_b += coord_b * cstrides[ord[j]]
          tmp_b //= cshape[ord[j]]
        bases[il] = base_b
    # threaded parabolic pass for this axis
    n0 = cshape[paxes[a]]
    s0 = cstrides[paxes[a]]
    if return_features:
      with nogil:
        if use_u32_feat:
          _nd_expand_parabolic_bases[uint32_t](distp, bases, lines, n0, s0, canis[paxes[a]], bb, fprev_u32, fnext_u32, parallel)
        else:
          _nd_expand_parabolic_bases[size_t](distp, bases, lines, n0, s0, canis[paxes[a]], bb, fprev_sz, fnext_sz, parallel)
      # swap features
      tmpF = feat_prev; feat_prev = feat_next; feat_next = tmpF
      if use_u32_feat:
        fprev_u32 = <uint32_t*> np.PyArray_DATA(feat_prev)
        fnext_u32 = <uint32_t*> np.PyArray_DATA(feat_next)
      else:
        fprev_sz = <size_t*> np.PyArray_DATA(feat_prev)
        fnext_sz = <size_t*> np.PyArray_DATA(feat_next)
    else:
      with nogil:
        _nd_expand_parabolic_labels_bases(distp, bases, lines, n0, s0, canis[paxes[a]], bb, lprev, lnext, parallel)
      # swap labels
      tmpL = lab_prev; lab_prev = lab_next; lab_next = tmpL
      lprev = <uint32_t*> np.PyArray_DATA(lab_prev)
      lnext = <uint32_t*> np.PyArray_DATA(lab_next)

  free(bases); free(ord); free(cshape); free(cstrides); free(paxes); free(canis)
  if return_features:
    # Map features to label values and return both
    out_flat = np.empty((total,), dtype=np.uint32)
    outp = <np.uint32_t*> np.PyArray_DATA(out_flat)
    labp2 = <np.uint32_t*> np.PyArray_DATA(labels_flat)
    if use_u32_feat:
      for il in range(total):
        idx2 = fprev_u32[il]
        outp[il] = labp2[idx2]
    else:
      for il in range(total):
        idx2 = fprev_sz[il]
        outp[il] = labp2[idx2]
    return out_flat.reshape(arr.shape), feat_prev.reshape(arr.shape)
  else:
    return lab_prev.reshape(arr.shape)


@cython.binding(True)
def feature_transform_nd(
    data, anisotropy=None, black_border=False,
    int parallel=1, voxel_graph=None,
    return_distances=False, features_dtype='auto'
  ):
  """ND feature transform (nearest seed) with optional squared distances.

  Parameters
  ----------
  data : ndarray
      Seed image (nonzero are seeds). Any numeric dtype accepted.
  anisotropy : float or sequence of float, optional
      Per-axis voxel size (default 1.0 for all axes).
  black_border : bool, optional
      If True, treat the border as background (default False).
  parallel : int, optional
      Number of threads; if <= 0, uses cpu_count().
  voxel_graph : ignored
      Present for API parity; not used in ND path.
  return_distances : bool, optional
      If True, also return squared EDT of the seed mask.
  features_dtype : {'auto','uint32','u32','uintp'}, optional
      Output dtype for feature indices; 'auto' picks uint32 if fits, else uintp.

  Returns
  -------
  feat : ndarray of uint32 or uintp
      Linear index of nearest seed for each voxel.
  dist : ndarray of float32, optional
      Squared Euclidean distance, if return_distances=True.
  """
  arr = np.asarray(data)
  if arr.size == 0:
    if return_distances:
      return np.zeros_like(arr, dtype=np.uintp), np.zeros_like(arr, dtype=np.float32)
    return np.zeros_like(arr, dtype=np.uintp)

  dims = arr.ndim
  if anisotropy is None:
    anis = (1.0,) * dims
  else:
    anis = tuple(anisotropy) if hasattr(anisotropy, '__len__') else (float(anisotropy),) * dims
    if len(anis) != dims:
      raise ValueError('anisotropy length must match data.ndim')
  if parallel <= 0:
    try:
      parallel = multiprocessing.cpu_count()
    except Exception:
      parallel = 1
  else:
    try:
      parallel = max(1, min(parallel, multiprocessing.cpu_count()))
    except Exception:
      parallel = max(1, parallel)

  # Use ND expand path with feature return on original array (nonzero=seeds)
  labels, feats = expand_labels_nd(arr.astype(np.uint32, copy=False), anis, black_border, parallel, voxel_graph, True)

  voxels = arr.size
  if isinstance(features_dtype, str):
    fd = features_dtype.lower()
    if fd in ('uint32', 'u32'):
      feats = feats.astype(np.uint32, copy=False)
    elif fd == 'uintp':
      feats = feats.astype(np.uintp, copy=False)
    else:
      feats = feats.astype(np.uint32 if voxels < 2**32 else np.uintp, copy=False)

  if return_distances:
    dist = edtsq_nd((arr != 0).astype(np.uint8, copy=False), anis, black_border, parallel, voxel_graph)
    return feats, dist
  return feats


def edt1d(data, anisotropy=1.0, native_bool black_border=False):
  result = edt1dsq(data, anisotropy, black_border)
  return np.sqrt(result, result)

def edt1dsq(data, anisotropy=1.0, native_bool black_border=False):
  cdef uint8_t[:] arr_memview8
  cdef uint16_t[:] arr_memview16
  cdef uint32_t[:] arr_memview32
  cdef uint64_t[:] arr_memview64
  cdef float[:] arr_memviewfloat
  cdef double[:] arr_memviewdouble

  cdef size_t voxels = data.size
  cdef np.ndarray[float, ndim=1] output = np.zeros( (voxels,), dtype=np.float32 )
  cdef float[:] outputview = output

  if data.dtype in (np.uint8, np.int8):
    arr_memview8 = data.astype(np.uint8)
    squared_edt_1d_multi_seg[uint8_t](
      <uint8_t*>&arr_memview8[0],
      &outputview[0],
      data.size,
      1,
      anisotropy,
      black_border
    )
  elif data.dtype in (np.uint16, np.int16):
    arr_memview16 = data.astype(np.uint16)
    squared_edt_1d_multi_seg[uint16_t](
      <uint16_t*>&arr_memview16[0],
      &outputview[0],
      data.size,
      1,
      anisotropy,
      black_border
    )
  elif data.dtype in (np.uint32, np.int32):
    arr_memview32 = data.astype(np.uint32)
    squared_edt_1d_multi_seg[uint32_t](
      <uint32_t*>&arr_memview32[0],
      &outputview[0],
      data.size,
      1,
      anisotropy,
      black_border
    )
  elif data.dtype in (np.uint64, np.int64):
    arr_memview64 = data.astype(np.uint64)
    squared_edt_1d_multi_seg[uint64_t](
      <uint64_t*>&arr_memview64[0],
      &outputview[0],
      data.size,
      1,
      anisotropy,
      black_border
    )
  elif data.dtype == np.float32:
    arr_memviewfloat = data
    squared_edt_1d_multi_seg[float](
      <float*>&arr_memviewfloat[0],
      &outputview[0],
      data.size,
      1,
      anisotropy,
      black_border
    )
  elif data.dtype == np.float64:
    arr_memviewdouble = data
    squared_edt_1d_multi_seg[double](
      <double*>&arr_memviewdouble[0],
      &outputview[0],
      data.size,
      1,
      anisotropy,
      black_border
    )
  elif data.dtype == bool:
    arr_memview8 = data.astype(np.uint8)
    squared_edt_1d_multi_seg[native_bool](
      <native_bool*>&arr_memview8[0],
      &outputview[0],
      data.size,
      1,
      anisotropy,
      black_border
    )
  
  return output

def edt2d(
    data, anisotropy=(1.0, 1.0), 
    native_bool black_border=False,
    parallel=1, voxel_graph=None
  ):
  result = edt2dsq(data, anisotropy, black_border, parallel, voxel_graph)
  return np.sqrt(result, result)

def edt2dsq(
    data, anisotropy=(1.0, 1.0), 
    native_bool black_border=False,
    parallel=1, voxel_graph=None
  ):
  if voxel_graph is not None:
    return __edt2dsq_voxel_graph(data, voxel_graph, anisotropy, black_border)
  return __edt2dsq(data, anisotropy, black_border, parallel)

def __edt2dsq(
    data, anisotropy=(1.0, 1.0), 
    native_bool black_border=False,
    parallel=1
  ):
  cdef uint8_t[:,:] arr_memview8
  cdef uint16_t[:,:] arr_memview16
  cdef uint32_t[:,:] arr_memview32
  cdef uint64_t[:,:] arr_memview64
  cdef float[:,:] arr_memviewfloat
  cdef double[:,:] arr_memviewdouble
  cdef native_bool[:,:] arr_memviewbool

  cdef size_t sx = data.shape[1] # C: rows
  cdef size_t sy = data.shape[0] # C: cols
  cdef float ax = anisotropy[1]
  cdef float ay = anisotropy[0]

  order = 'C'
  if data.flags.f_contiguous:
    sx = data.shape[0] # F: cols
    sy = data.shape[1] # F: rows
    ax = anisotropy[0]
    ay = anisotropy[1]
    order = 'F'

  cdef size_t voxels = sx * sy
  cdef np.ndarray[float, ndim=1] output = np.zeros( (voxels,), dtype=np.float32 )
  cdef float[:] outputview = output

  if data.dtype in (np.uint8, np.int8):
    arr_memview8 = data.astype(np.uint8)
    _edt2dsq_with_features[uint8_t, size_t](
      <uint8_t*>&arr_memview8[0,0],
      sx, sy,
      ax, ay,
      black_border, parallel,
      &outputview[0], NULL
    )
  elif data.dtype in (np.uint16, np.int16):
    arr_memview16 = data.astype(np.uint16)
    _edt2dsq_with_features[uint16_t, size_t](
      <uint16_t*>&arr_memview16[0,0],
      sx, sy,
      ax, ay,
      black_border, parallel,
      &outputview[0], NULL
    )
  elif data.dtype in (np.uint32, np.int32):
    arr_memview32 = data.astype(np.uint32)
    _edt2dsq_with_features[uint32_t, size_t](
      <uint32_t*>&arr_memview32[0,0],
      sx, sy,
      ax, ay,
      black_border, parallel,
      &outputview[0], NULL
    )
  elif data.dtype in (np.uint64, np.int64):
    arr_memview64 = data.astype(np.uint64)
    _edt2dsq_with_features[uint64_t, size_t](
      <uint64_t*>&arr_memview64[0,0],
      sx, sy,
      ax, ay,
      black_border, parallel,
      &outputview[0], NULL
    )
  elif data.dtype == np.float32:
    arr_memviewfloat = data
    _edt2dsq_with_features[float, size_t](
      <float*>&arr_memviewfloat[0,0],
      sx, sy,
      ax, ay,
      black_border, parallel,
      &outputview[0], NULL
    )
  elif data.dtype == np.float64:
    arr_memviewdouble = data
    _edt2dsq_with_features[double, size_t](
      <double*>&arr_memviewdouble[0,0],
      sx, sy,
      ax, ay,
      black_border, parallel,
      &outputview[0], NULL
    )
  elif data.dtype == bool:
    arr_memview8 = data.view(np.uint8)
    _edt2dsq_with_features[native_bool, size_t](
      <native_bool*>&arr_memview8[0,0],
      sx, sy,
      ax, ay,
      black_border, parallel,
      &outputview[0], NULL
    )

  return output.reshape(data.shape, order=order)

def __edt2dsq_voxel_graph(
    data, voxel_graph, anisotropy=(1.0, 1.0), 
    native_bool black_border=False,
  ):
  cdef uint8_t[:,:] arr_memview8
  cdef uint16_t[:,:] arr_memview16
  cdef uint32_t[:,:] arr_memview32
  cdef uint64_t[:,:] arr_memview64
  cdef float[:,:] arr_memviewfloat
  cdef double[:,:] arr_memviewdouble
  cdef native_bool[:,:] arr_memviewbool

  cdef uint8_t[:,:] graph_memview8
  if voxel_graph.dtype in (np.uint8, np.int8):
    graph_memview8 = voxel_graph.view(np.uint8)
  else:
    graph_memview8 = voxel_graph.astype(np.uint8) # we only need first 6 bits

  cdef size_t sx = data.shape[1] # C: rows
  cdef size_t sy = data.shape[0] # C: cols
  cdef float ax = anisotropy[1]
  cdef float ay = anisotropy[0]
  order = 'C'

  if data.flags.f_contiguous:
    sx = data.shape[0] # F: cols
    sy = data.shape[1] # F: rows
    ax = anisotropy[0]
    ay = anisotropy[1]
    order = 'F'

  cdef size_t voxels = sx * sy
  cdef np.ndarray[float, ndim=1] output = np.zeros( (voxels,), dtype=np.float32 )
  cdef float[:] outputview = output

  if data.dtype in (np.uint8, np.int8):
    arr_memview8 = data.astype(np.uint8)
    _edt2dsq_voxel_graph[uint8_t,uint8_t](
      <uint8_t*>&arr_memview8[0,0],
      <uint8_t*>&graph_memview8[0,0],
      sx, sy,
      ax, ay,
      black_border,
      &outputview[0]
    )
  elif data.dtype in (np.uint16, np.int16):
    arr_memview16 = data.astype(np.uint16)
    _edt2dsq_voxel_graph[uint16_t,uint8_t](
      <uint16_t*>&arr_memview16[0,0],
      <uint8_t*>&graph_memview8[0,0],
      sx, sy,
      ax, ay,
      black_border,
      &outputview[0]      
    )
  elif data.dtype in (np.uint32, np.int32):
    arr_memview32 = data.astype(np.uint32)
    _edt2dsq_voxel_graph[uint32_t,uint8_t](
      <uint32_t*>&arr_memview32[0,0],
      <uint8_t*>&graph_memview8[0,0],
      sx, sy,
      ax, ay,
      black_border,
      &outputview[0]      
    )
  elif data.dtype in (np.uint64, np.int64):
    arr_memview64 = data.astype(np.uint64)
    _edt2dsq_voxel_graph[uint64_t,uint8_t](
      <uint64_t*>&arr_memview64[0,0],
      <uint8_t*>&graph_memview8[0,0],
      sx, sy,
      ax, ay,
      black_border,
      &outputview[0]      
    )
  elif data.dtype == np.float32:
    arr_memviewfloat = data
    _edt2dsq_voxel_graph[float,uint8_t](
      <float*>&arr_memviewfloat[0,0],
      <uint8_t*>&graph_memview8[0,0],
      sx, sy,
      ax, ay,
      black_border,
      &outputview[0]      
    )
  elif data.dtype == np.float64:
    arr_memviewdouble = data
    _edt2dsq_voxel_graph[double,uint8_t](
      <double*>&arr_memviewdouble[0,0],
      <uint8_t*>&graph_memview8[0,0],
      sx, sy,
      ax, ay,
      black_border,
      &outputview[0]      
    )
  elif data.dtype == bool:
    arr_memview8 = data.view(np.uint8)
    _edt2dsq_voxel_graph[native_bool,uint8_t](
      <native_bool*>&arr_memview8[0,0],
      <uint8_t*>&graph_memview8[0,0],
      sx, sy,
      ax, ay,
      black_border,
      &outputview[0]      
    )

  return output.reshape( data.shape, order=order)

@cython.binding(True)
def edtsq_nd(
  data, anisotropy=None, native_bool black_border=False,
  int parallel=1, voxel_graph=None, order=None,
):
  """Genuine ND squared EDT using 1D kernels, matching 2D/3D semantics.

  Algorithm:
    - Axis 0 (x-fastest for C, first for F): multi-seg 1D on labels -> workspace floats (stride=1)
    - tofinite if !black_border
    - Axes 1..N-1: parabolic multi-seg 1D on labels with stride along that axis, writing to workspace
    - toinfinite at end if !black_border
  """
  if isinstance(data, list):
    data = np.array(data)
  arr = np.asarray(data)
  if arr.size == 0:
    return np.zeros_like(arr, dtype=np.float32)
  # ensure contiguous buffer
  if not arr.flags.c_contiguous and not arr.flags.f_contiguous:
    arr = np.ascontiguousarray(arr)

  dims = arr.ndim
  # anisotropy vector
  if anisotropy is None:
    anis = (1.0,) * dims
  else:
    anis = tuple(anisotropy) if hasattr(anisotropy, '__len__') else (float(anisotropy),) * dims
    if len(anis) != dims:
      raise ValueError('anisotropy length must match data.ndim')

  # Declarations for optional 1D fast-path
  cdef uint8_t[:] arr1_u8
  cdef uint16_t[:] arr1_u16
  cdef uint32_t[:] arr1_u32
  cdef uint64_t[:] arr1_u64
  cdef float[:] arr1_f32
  cdef double[:] arr1_f64
  cdef np.ndarray[float, ndim=1] out1
  cdef float[:] out1_view

  # Fast-path: 1D do inline specialized kernel (no external call)
  if dims == 1:
    out1 = np.zeros((arr.size,), dtype=np.float32)
    out1_view = out1
    if arr.dtype in (np.uint8, np.int8):
      arr1_u8 = arr.astype(np.uint8, copy=False)
      squared_edt_1d_multi_seg[uint8_t](<uint8_t*>&arr1_u8[0], &out1_view[0], arr.size, 1, <float>anis[0], black_border)
    elif arr.dtype in (np.uint16, np.int16):
      arr1_u16 = arr.astype(np.uint16, copy=False)
      squared_edt_1d_multi_seg[uint16_t](<uint16_t*>&arr1_u16[0], &out1_view[0], arr.size, 1, <float>anis[0], black_border)
    elif arr.dtype in (np.uint32, np.int32):
      arr1_u32 = arr.astype(np.uint32, copy=False)
      squared_edt_1d_multi_seg[uint32_t](<uint32_t*>&arr1_u32[0], &out1_view[0], arr.size, 1, <float>anis[0], black_border)
    elif arr.dtype in (np.uint64, np.int64):
      arr1_u64 = arr.astype(np.uint64, copy=False)
      squared_edt_1d_multi_seg[uint64_t](<uint64_t*>&arr1_u64[0], &out1_view[0], arr.size, 1, <float>anis[0], black_border)
    elif arr.dtype == np.float32:
      arr1_f32 = arr
      squared_edt_1d_multi_seg[float](<float*>&arr1_f32[0], &out1_view[0], arr.size, 1, <float>anis[0], black_border)
    elif arr.dtype == np.float64:
      arr1_f64 = arr
      squared_edt_1d_multi_seg[double](<double*>&arr1_f64[0], &out1_view[0], arr.size, 1, <float>anis[0], black_border)
    elif arr.dtype == bool:
      arr1_u8 = arr.astype(np.uint8, copy=False)
      squared_edt_1d_multi_seg[native_bool](<native_bool*>&arr1_u8[0], &out1_view[0], arr.size, 1, <float>anis[0], black_border)
    return out1.reshape(arr.shape)

  # voxel_graph is only defined for 2D/3D specialized APIs; ND core ignores it.
  if voxel_graph is not None:
    raise TypeError('voxel_graph is only supported by 2D/3D specialized APIs')

  # internal axis order: x-fastest axis first
  c_order = arr.flags.c_contiguous
  axes = list(range(dims-1, -1, -1)) if c_order else list(range(dims))

  # logical strides (elements)
  shape = tuple(arr.shape)
  strides = [1] * dims
  if c_order:
    for i in range(dims-2, -1, -1):
      strides[i] = strides[i+1] * shape[i+1]
  else:
    for i in range(1, dims):
      strides[i] = strides[i-1] * shape[i-1]

  total = int(arr.size)
  out = np.zeros((total,), dtype=np.float32)

  # dtype dispatch: get flat pointers
  flat = arr.ravel(order='K')
  cdef float* outp = <float*> np.PyArray_DATA(out)
  
  # Build C arrays for loop indices and anisotropy
  cdef Py_ssize_t nd = dims
  cdef size_t* cshape = <size_t*> malloc(nd * sizeof(size_t))
  cdef size_t* cstrides = <size_t*> malloc(nd * sizeof(size_t))
  cdef Py_ssize_t* caxes = <Py_ssize_t*> malloc(nd * sizeof(Py_ssize_t))
  cdef Py_ssize_t* paxes = <Py_ssize_t*> malloc(nd * sizeof(Py_ssize_t))
  cdef float* canis = <float*> malloc(nd * sizeof(float))
  if cshape == NULL or cstrides == NULL or caxes == NULL or paxes == NULL or canis == NULL:
    if cshape != NULL: free(cshape)
    if cstrides != NULL: free(cstrides)
    if caxes != NULL: free(caxes)
    if paxes != NULL: free(paxes)
    if canis != NULL: free(canis)
    raise MemoryError('Allocation failure in edtsq_nd')
  cdef Py_ssize_t ii
  for ii in range(nd):
    cshape[ii] = <size_t>shape[ii]
    cstrides[ii] = <size_t>strides[ii]
    caxes[ii] = <Py_ssize_t>axes[ii]
    canis[ii] = <float>anis[ii]

  # Determine pass axis order: sort axes by increasing stride (cache locality)
  for ii in range(nd):
    paxes[ii] = ii
  cdef Py_ssize_t ii2
  cdef Py_ssize_t keyi
  cdef Py_ssize_t jj
  for ii2 in range(1, nd):
    keyi = paxes[ii2]
    jj = ii2 - 1
    while jj >= 0 and (
      cstrides[paxes[jj]] > cstrides[keyi]
      or (cstrides[paxes[jj]] == cstrides[keyi] and cshape[paxes[jj]] < cshape[keyi])
    ):
      paxes[jj+1] = paxes[jj]
      jj -= 1
    paxes[jj+1] = keyi

  cdef void* dataptr = <void*> np.PyArray_DATA(flat)
  cdef size_t tot = <size_t> total
  cdef size_t kk
  cdef size_t kk2
  cdef size_t lines
  # Max number of lines among all passes; used to size bases buffer safely.
  cdef size_t max_lines = 0
  for ii2 in range(nd):
    lines = tot // <size_t>cshape[paxes[ii2]]
    if lines > max_lines:
      max_lines = lines
  cdef size_t* bases
  cdef Py_ssize_t a
  cdef size_t il
  cdef size_t base_b
  cdef size_t tmp_b
  cdef size_t coord_b
  cdef Py_ssize_t ord_len
  cdef Py_ssize_t* ord
  cdef Py_ssize_t pos
  cdef Py_ssize_t j
  cdef Py_ssize_t key_ax
  if arr.dtype in (np.uint8, np.int8):
    # Precompute bases for axis 0 and run threaded multi pass
    lines = tot // <size_t>cshape[paxes[0]]
    bases = <size_t*> malloc(max_lines * sizeof(size_t))
    if bases == NULL:
      free(cshape); free(cstrides); free(caxes); free(paxes); free(canis)
      raise MemoryError('Allocation failure for bases')
    # Build axis order for base enumeration: other axes sorted by increasing stride
    ord_len = nd - 1
    ord = <Py_ssize_t*> malloc(ord_len * sizeof(Py_ssize_t))
    if ord == NULL:
      free(bases); free(cshape); free(cstrides); free(caxes); free(canis)
      raise MemoryError('Allocation failure for ord')
    # Fill with axes except caxes[0]
    pos = 0
    for ii in range(nd):
      if ii != paxes[0]:
        ord[pos] = ii
        pos += 1
    # Simple insertion sort by stride (ascending)
    for il in range(1, ord_len):
      key_ax = ord[il]
      j = il - 1
      while j >= 0 and (
        cstrides[ord[j]] > cstrides[key_ax]
        or (cstrides[ord[j]] == cstrides[key_ax] and cshape[ord[j]] < cshape[key_ax])
      ):
        ord[j+1] = ord[j]
        j -= 1
      ord[j+1] = key_ax
    with nogil:
      for il in range(lines):
        base_b = 0
        tmp_b = il
        for j in range(ord_len):
          coord_b = tmp_b % cshape[ord[j]]
          base_b += coord_b * cstrides[ord[j]]
          tmp_b //= cshape[ord[j]]
        bases[il] = base_b
      _nd_pass_multi_bases[uint8_t](<uint8_t*>dataptr, outp, bases, lines, <size_t>cshape[paxes[0]], <size_t>cstrides[paxes[0]], canis[paxes[0]], black_border, parallel)
    if not black_border:
      with nogil:
        for kk in range(tot):
          if isinf(outp[kk]):
            outp[kk] = 3.4028234e+38
    for a in range(1, nd):
      lines = tot // <size_t>cshape[paxes[a]]
      with nogil:
        for kk in range(tot):
          pass
      # reuse bases buffer
      if lines * sizeof(size_t) > 0:
        # recompute bases for this axis using stride-ascending order of other axes
        # build ord for this axis
        ord_len = nd - 1
        free(ord)
        ord = <Py_ssize_t*> malloc(ord_len * sizeof(Py_ssize_t))
        if ord == NULL:
          free(bases); free(cshape); free(cstrides); free(caxes); free(canis)
          raise MemoryError('Allocation failure for ord')
        pos = 0
        for ii in range(nd):
          if ii != paxes[a]:
            ord[pos] = ii
            pos += 1
        for il in range(1, ord_len):
          key_ax = ord[il]
          j = il - 1
          while j >= 0 and (
            cstrides[ord[j]] > cstrides[key_ax]
            or (cstrides[ord[j]] == cstrides[key_ax] and cshape[ord[j]] < cshape[key_ax])
          ):
            ord[j+1] = ord[j]
            j -= 1
          ord[j+1] = key_ax
        with nogil:
          for il in range(lines):
            base_b = 0
            tmp_b = il
            for j in range(ord_len):
              coord_b = tmp_b % cshape[ord[j]]
              base_b += cstrides[ord[j]] * coord_b
              tmp_b //= cshape[ord[j]]
            bases[il] = base_b
        with nogil:
          _nd_pass_parabolic_bases[uint8_t](<uint8_t*>dataptr, outp, bases, lines, <size_t>cshape[paxes[a]], <size_t>cstrides[paxes[a]], canis[paxes[a]], black_border, parallel)
    free(bases); free(ord)
    if not black_border:
      with nogil:
        for kk2 in range(tot):
          if outp[kk2] >= 3.4028234e+38:
            outp[kk2] = INFINITY
  elif arr.dtype in (np.uint16, np.int16):
    lines = tot // <size_t>cshape[paxes[0]]
    bases = <size_t*> malloc(max_lines * sizeof(size_t))
    if bases == NULL:
      free(cshape); free(cstrides); free(caxes); free(canis)
      raise MemoryError('Allocation failure for bases')
    # Build stride-ascending order of other axes for axis 0
    ord_len = nd - 1
    ord = <Py_ssize_t*> malloc(ord_len * sizeof(Py_ssize_t))
    if ord == NULL:
      free(bases); free(cshape); free(cstrides); free(caxes); free(paxes); free(canis)
      raise MemoryError('Allocation failure for ord')
    pos = 0
    for ii in range(nd):
      if ii != paxes[0]:
        ord[pos] = ii
        pos += 1
    for il in range(1, ord_len):
      key_ax = ord[il]
      j = il - 1
      while j >= 0 and (
        cstrides[ord[j]] > cstrides[key_ax]
        or (cstrides[ord[j]] == cstrides[key_ax] and cshape[ord[j]] < cshape[key_ax])
      ):
        ord[j+1] = ord[j]
        j -= 1
      ord[j+1] = key_ax
    with nogil:
      for il in range(lines):
        base_b = 0
        tmp_b = il
        for j in range(ord_len):
          coord_b = tmp_b % cshape[ord[j]]
          base_b += coord_b * cstrides[ord[j]]
          tmp_b //= cshape[ord[j]]
        bases[il] = base_b
      _nd_pass_multi_bases[uint16_t](<uint16_t*>dataptr, outp, bases, lines, <size_t>cshape[paxes[0]], <size_t>cstrides[paxes[0]], canis[paxes[0]], black_border, parallel)
    if not black_border:
      with nogil:
        for kk in range(tot):
          if isinf(outp[kk]):
            outp[kk] = 3.4028234e+38
    for a in range(1, nd):
      lines = tot // <size_t>cshape[paxes[a]]
      # Build ord for this axis
      free(ord)
      ord = <Py_ssize_t*> malloc((nd-1) * sizeof(Py_ssize_t))
      if ord == NULL:
        free(bases); free(cshape); free(cstrides); free(caxes); free(canis)
        raise MemoryError('Allocation failure for ord')
      pos = 0
      for ii in range(nd):
        if ii != paxes[a]:
          ord[pos] = ii
          pos += 1
      for il in range(1, nd-1):
        key_ax = ord[il]
        j = il - 1
        while j >= 0 and (
          cstrides[ord[j]] > cstrides[key_ax]
          or (cstrides[ord[j]] == cstrides[key_ax] and cshape[ord[j]] < cshape[key_ax])
        ):
          ord[j+1] = ord[j]
          j -= 1
        ord[j+1] = key_ax
      with nogil:
        for il in range(lines):
          base_b = 0
          tmp_b = il
          for j in range(nd-1):
            coord_b = tmp_b % cshape[ord[j]]
            base_b += coord_b * cstrides[ord[j]]
            tmp_b //= cshape[ord[j]]
          bases[il] = base_b
      with nogil:
        _nd_pass_parabolic_bases[uint16_t](<uint16_t*>dataptr, outp, bases, lines, <size_t>cshape[paxes[a]], <size_t>cstrides[paxes[a]], canis[paxes[a]], black_border, parallel)
    free(bases); free(ord)
    if not black_border:
      with nogil:
        for kk2 in range(tot):
          if outp[kk2] >= 3.4028234e+38:
            outp[kk2] = INFINITY
  elif arr.dtype in (np.uint32, np.int32):
    lines = tot // <size_t>cshape[paxes[0]]
    bases = <size_t*> malloc(max_lines * sizeof(size_t))
    if bases == NULL:
      free(cshape); free(cstrides); free(caxes); free(canis)
      raise MemoryError('Allocation failure for bases')
    ord_len = nd - 1
    ord = <Py_ssize_t*> malloc(ord_len * sizeof(Py_ssize_t))
    if ord == NULL:
      free(bases); free(cshape); free(cstrides); free(caxes); free(paxes); free(canis)
      raise MemoryError('Allocation failure for ord')
    pos = 0
    for ii in range(nd):
      if ii != paxes[0]:
        ord[pos] = ii
        pos += 1
    for il in range(1, ord_len):
      key_ax = ord[il]
      j = il - 1
      while j >= 0 and (
        cstrides[ord[j]] > cstrides[key_ax]
        or (cstrides[ord[j]] == cstrides[key_ax] and cshape[ord[j]] < cshape[key_ax])
      ):
        ord[j+1] = ord[j]
        j -= 1
      ord[j+1] = key_ax
    with nogil:
      for il in range(lines):
        base_b = 0
        tmp_b = il
        for j in range(ord_len):
          coord_b = tmp_b % cshape[ord[j]]
          base_b += coord_b * cstrides[ord[j]]
          tmp_b //= cshape[ord[j]]
        bases[il] = base_b
      _nd_pass_multi_bases[uint32_t](<uint32_t*>dataptr, outp, bases, lines, <size_t>cshape[paxes[0]], <size_t>cstrides[paxes[0]], canis[paxes[0]], black_border, parallel)
    if not black_border:
      with nogil:
        for kk in range(tot):
          if isinf(outp[kk]):
            outp[kk] = 3.4028234e+38
    for a in range(1, nd):
      lines = tot // <size_t>cshape[paxes[a]]
      free(ord)
      ord = <Py_ssize_t*> malloc((nd-1) * sizeof(Py_ssize_t))
      if ord == NULL:
        free(bases); free(cshape); free(cstrides); free(caxes); free(canis)
        raise MemoryError('Allocation failure for ord')
      pos = 0
      for ii in range(nd):
        if ii != paxes[a]:
          ord[pos] = ii
          pos += 1
      for il in range(1, nd-1):
        key_ax = ord[il]
        j = il - 1
        while j >= 0 and (
          cstrides[ord[j]] > cstrides[key_ax]
          or (cstrides[ord[j]] == cstrides[key_ax] and cshape[ord[j]] < cshape[key_ax])
        ):
          ord[j+1] = ord[j]
          j -= 1
        ord[j+1] = key_ax
      with nogil:
        for il in range(lines):
          base_b = 0
          tmp_b = il
          for j in range(nd-1):
            coord_b = tmp_b % cshape[ord[j]]
            base_b += coord_b * cstrides[ord[j]]
            tmp_b //= cshape[ord[j]]
          bases[il] = base_b
      with nogil:
        _nd_pass_parabolic_bases[uint32_t](<uint32_t*>dataptr, outp, bases, lines, <size_t>cshape[paxes[a]], <size_t>cstrides[paxes[a]], canis[paxes[a]], black_border, parallel)
    free(bases); free(ord)
    if not black_border:
      with nogil:
        for kk2 in range(tot):
          if outp[kk2] >= 3.4028234e+38:
            outp[kk2] = INFINITY
  elif arr.dtype in (np.uint64, np.int64):
    lines = tot // <size_t>cshape[paxes[0]]
    bases = <size_t*> malloc(max_lines * sizeof(size_t))
    if bases == NULL:
      free(cshape); free(cstrides); free(caxes); free(canis)
      raise MemoryError('Allocation failure for bases')
    ord_len = nd - 1
    ord = <Py_ssize_t*> malloc(ord_len * sizeof(Py_ssize_t))
    if ord == NULL:
      free(bases); free(cshape); free(cstrides); free(caxes); free(paxes); free(canis)
      raise MemoryError('Allocation failure for ord')
    pos = 0
    for ii in range(nd):
      if ii != paxes[0]:
        ord[pos] = ii
        pos += 1
    for il in range(1, ord_len):
      key_ax = ord[il]
      j = il - 1
      while j >= 0 and (
        cstrides[ord[j]] > cstrides[key_ax]
        or (cstrides[ord[j]] == cstrides[key_ax] and cshape[ord[j]] < cshape[key_ax])
      ):
        ord[j+1] = ord[j]
        j -= 1
      ord[j+1] = key_ax
    with nogil:
      for il in range(lines):
        base_b = 0
        tmp_b = il
        for j in range(ord_len):
          coord_b = tmp_b % cshape[ord[j]]
          base_b += coord_b * cstrides[ord[j]]
          tmp_b //= cshape[ord[j]]
        bases[il] = base_b
      _nd_pass_multi_bases[uint64_t](<uint64_t*>dataptr, outp, bases, lines, <size_t>cshape[paxes[0]], <size_t>cstrides[paxes[0]], canis[paxes[0]], black_border, parallel)
    if not black_border:
      with nogil:
        for kk in range(tot):
          if isinf(outp[kk]):
            outp[kk] = 3.4028234e+38
    for a in range(1, nd):
      lines = tot // <size_t>cshape[paxes[a]]
      free(ord)
      ord = <Py_ssize_t*> malloc((nd-1) * sizeof(Py_ssize_t))
      if ord == NULL:
        free(bases); free(cshape); free(cstrides); free(caxes); free(canis)
        raise MemoryError('Allocation failure for ord')
      pos = 0
      for ii in range(nd):
        if ii != paxes[a]:
          ord[pos] = ii
          pos += 1
      for il in range(1, nd-1):
        key_ax = ord[il]
        j = il - 1
        while j >= 0 and (
          cstrides[ord[j]] > cstrides[key_ax]
          or (cstrides[ord[j]] == cstrides[key_ax] and cshape[ord[j]] < cshape[key_ax])
        ):
          ord[j+1] = ord[j]
          j -= 1
        ord[j+1] = key_ax
      with nogil:
        for il in range(lines):
          base_b = 0
          tmp_b = il
          for j in range(nd-1):
            coord_b = tmp_b % cshape[ord[j]]
            base_b += coord_b * cstrides[ord[j]]
            tmp_b //= cshape[ord[j]]
          bases[il] = base_b
      with nogil:
        _nd_pass_parabolic_bases[uint64_t](<uint64_t*>dataptr, outp, bases, lines, <size_t>cshape[paxes[a]], <size_t>cstrides[paxes[a]], canis[paxes[a]], black_border, parallel)
    free(bases); free(ord)
    if not black_border:
      with nogil:
        for kk2 in range(tot):
          if outp[kk2] >= 3.4028234e+38:
            outp[kk2] = INFINITY
  elif arr.dtype == np.float32:
    lines = tot // <size_t>cshape[paxes[0]]
    bases = <size_t*> malloc(lines * sizeof(size_t))
    if bases == NULL:
      free(cshape); free(cstrides); free(caxes); free(canis)
      raise MemoryError('Allocation failure for bases')
    ord_len = nd - 1
    ord = <Py_ssize_t*> malloc(ord_len * sizeof(Py_ssize_t))
    if ord == NULL:
      free(bases); free(cshape); free(cstrides); free(caxes); free(paxes); free(canis)
      raise MemoryError('Allocation failure for ord')
    pos = 0
    for ii in range(nd):
      if ii != paxes[0]:
        ord[pos] = ii
        pos += 1
    for il in range(1, ord_len):
      key_ax = ord[il]
      j = il - 1
      while j >= 0 and (
        cstrides[ord[j]] > cstrides[key_ax]
        or (cstrides[ord[j]] == cstrides[key_ax] and cshape[ord[j]] < cshape[key_ax])
      ):
        ord[j+1] = ord[j]
        j -= 1
      ord[j+1] = key_ax
    with nogil:
      for il in range(lines):
        base_b = 0
        tmp_b = il
        for j in range(ord_len):
          coord_b = tmp_b % cshape[ord[j]]
          base_b += coord_b * cstrides[ord[j]]
          tmp_b //= cshape[ord[j]]
        bases[il] = base_b
      _nd_pass_multi_bases[float](<float*>dataptr, outp, bases, lines, <size_t>cshape[paxes[0]], <size_t>cstrides[paxes[0]], canis[paxes[0]], black_border, parallel)
    if not black_border:
      with nogil:
        for kk in range(tot):
          if isinf(outp[kk]):
            outp[kk] = 3.4028234e+38
    for a in range(1, nd):
      lines = tot // <size_t>cshape[paxes[a]]
      free(ord)
      ord = <Py_ssize_t*> malloc((nd-1) * sizeof(Py_ssize_t))
      if ord == NULL:
        free(bases); free(cshape); free(cstrides); free(caxes); free(canis)
        raise MemoryError('Allocation failure for ord')
      pos = 0
      for ii in range(nd):
        if ii != paxes[a]:
          ord[pos] = ii
          pos += 1
      for il in range(1, nd-1):
        key_ax = ord[il]
        j = il - 1
        while j >= 0 and (
          cstrides[ord[j]] > cstrides[key_ax]
          or (cstrides[ord[j]] == cstrides[key_ax] and cshape[ord[j]] < cshape[key_ax])
        ):
          ord[j+1] = ord[j]
          j -= 1
        ord[j+1] = key_ax
      with nogil:
        for il in range(lines):
          base_b = 0
          tmp_b = il
          for j in range(nd-1):
            coord_b = tmp_b % cshape[ord[j]]
            base_b += coord_b * cstrides[ord[j]]
            tmp_b //= cshape[ord[j]]
          bases[il] = base_b
      with nogil:
        _nd_pass_parabolic_bases[float](<float*>dataptr, outp, bases, lines, <size_t>cshape[paxes[a]], <size_t>cstrides[paxes[a]], canis[paxes[a]], black_border, parallel)
    free(bases); free(ord)
    if not black_border:
      with nogil:
        for kk2 in range(tot):
          if outp[kk2] >= 3.4028234e+38:
            outp[kk2] = INFINITY
  elif arr.dtype == np.float64:
    lines = tot // <size_t>cshape[paxes[0]]
    bases = <size_t*> malloc(lines * sizeof(size_t))
    if bases == NULL:
      free(cshape); free(cstrides); free(caxes); free(canis)
      raise MemoryError('Allocation failure for bases')
    ord_len = nd - 1
    ord = <Py_ssize_t*> malloc(ord_len * sizeof(Py_ssize_t))
    if ord == NULL:
      free(bases); free(cshape); free(cstrides); free(caxes); free(paxes); free(canis)
      raise MemoryError('Allocation failure for ord')
    pos = 0
    for ii in range(nd):
      if ii != paxes[0]:
        ord[pos] = ii
        pos += 1
    for il in range(1, ord_len):
      key_ax = ord[il]
      j = il - 1
      while j >= 0 and (
        cstrides[ord[j]] > cstrides[key_ax]
        or (cstrides[ord[j]] == cstrides[key_ax] and cshape[ord[j]] < cshape[key_ax])
      ):
        ord[j+1] = ord[j]
        j -= 1
      ord[j+1] = key_ax
    with nogil:
      for il in range(lines):
        base_b = 0
        tmp_b = il
        for j in range(ord_len):
          coord_b = tmp_b % cshape[ord[j]]
          base_b += coord_b * cstrides[ord[j]]
          tmp_b //= cshape[ord[j]]
        bases[il] = base_b
      _nd_pass_multi_bases[double](<double*>dataptr, outp, bases, lines, <size_t>cshape[paxes[0]], <size_t>cstrides[paxes[0]], canis[paxes[0]], black_border, parallel)
    if not black_border:
      with nogil:
        for kk in range(tot):
          if isinf(outp[kk]):
            outp[kk] = 3.4028234e+38
    for a in range(1, nd):
      lines = tot // <size_t>cshape[paxes[a]]
      free(ord)
      ord = <Py_ssize_t*> malloc((nd-1) * sizeof(Py_ssize_t))
      if ord == NULL:
        free(bases); free(cshape); free(cstrides); free(caxes); free(canis)
        raise MemoryError('Allocation failure for ord')
      pos = 0
      for ii in range(nd):
        if ii != paxes[a]:
          ord[pos] = ii
          pos += 1
      for il in range(1, nd-1):
        key_ax = ord[il]
        j = il - 1
        while j >= 0 and (
          cstrides[ord[j]] > cstrides[key_ax]
          or (cstrides[ord[j]] == cstrides[key_ax] and cshape[ord[j]] < cshape[key_ax])
        ):
          ord[j+1] = ord[j]
          j -= 1
        ord[j+1] = key_ax
      with nogil:
        for il in range(lines):
          base_b = 0
          tmp_b = il
          for j in range(nd-1):
            coord_b = tmp_b % cshape[ord[j]]
            base_b += coord_b * cstrides[ord[j]]
            tmp_b //= cshape[ord[j]]
          bases[il] = base_b
      with nogil:
        _nd_pass_parabolic_bases[double](<double*>dataptr, outp, bases, lines, <size_t>cshape[paxes[a]], <size_t>cstrides[paxes[a]], canis[paxes[a]], black_border, parallel)
    free(bases); free(ord)
    if not black_border:
      with nogil:
        for kk2 in range(tot):
          if outp[kk2] >= 3.4028234e+38:
            outp[kk2] = INFINITY
  elif arr.dtype == bool:
    lines = tot // <size_t>cshape[paxes[0]]
    bases = <size_t*> malloc(lines * sizeof(size_t))
    if bases == NULL:
      free(cshape); free(cstrides); free(caxes); free(canis)
      raise MemoryError('Allocation failure for bases')
    ord_len = nd - 1
    ord = <Py_ssize_t*> malloc(ord_len * sizeof(Py_ssize_t))
    if ord == NULL:
      free(bases); free(cshape); free(cstrides); free(caxes); free(paxes); free(canis)
      raise MemoryError('Allocation failure for ord')
    pos = 0
    for ii in range(nd):
      if ii != paxes[0]:
        ord[pos] = ii
        pos += 1
    for il in range(1, ord_len):
      key_ax = ord[il]
      j = il - 1
      while j >= 0 and (
        cstrides[ord[j]] > cstrides[key_ax]
        or (cstrides[ord[j]] == cstrides[key_ax] and cshape[ord[j]] < cshape[key_ax])
      ):
        ord[j+1] = ord[j]
        j -= 1
      ord[j+1] = key_ax
    with nogil:
      for il in range(lines):
        base_b = 0
        tmp_b = il
        for j in range(ord_len):
          coord_b = tmp_b % cshape[ord[j]]
          base_b += coord_b * cstrides[ord[j]]
          tmp_b //= cshape[ord[j]]
        bases[il] = base_b
      _nd_pass_multi_bases[native_bool](<native_bool*>dataptr, outp, bases, lines, <size_t>cshape[paxes[0]], <size_t>cstrides[paxes[0]], canis[paxes[0]], black_border, parallel)
    if not black_border:
      with nogil:
        for kk in range(tot):
          if isinf(outp[kk]):
            outp[kk] = 3.4028234e+38
    for a in range(1, nd):
      lines = tot // <size_t>cshape[paxes[a]]
      free(ord)
      ord = <Py_ssize_t*> malloc((nd-1) * sizeof(Py_ssize_t))
      if ord == NULL:
        free(bases); free(cshape); free(cstrides); free(caxes); free(canis)
        raise MemoryError('Allocation failure for ord')
      pos = 0
      for ii in range(nd):
        if ii != paxes[a]:
          ord[pos] = ii
          pos += 1
      for il in range(1, nd-1):
        key_ax = ord[il]
        j = il - 1
        while j >= 0 and (
          cstrides[ord[j]] > cstrides[key_ax]
          or (cstrides[ord[j]] == cstrides[key_ax] and cshape[ord[j]] < cshape[key_ax])
        ):
          ord[j+1] = ord[j]
          j -= 1
        ord[j+1] = key_ax
      with nogil:
        for il in range(lines):
          base_b = 0
          tmp_b = il
          for j in range(nd-1):
            coord_b = tmp_b % cshape[ord[j]]
            base_b += coord_b * cstrides[ord[j]]
            tmp_b //= cshape[ord[j]]
          bases[il] = base_b
      with nogil:
        _nd_pass_parabolic_bases[native_bool](<native_bool*>dataptr, outp, bases, lines, <size_t>cshape[paxes[a]], <size_t>cstrides[paxes[a]], canis[paxes[a]], black_border, parallel)
    free(bases); free(ord)
    if not black_border:
      with nogil:
        for kk2 in range(tot):
          if outp[kk2] >= 3.4028234e+38:
            outp[kk2] = INFINITY
  else:
    free(cshape); free(cstrides); free(caxes); free(canis)
    raise TypeError('Unsupported dtype')

  free(cshape); free(cstrides); free(caxes); free(paxes); free(canis)
  return out.reshape(arr.shape, order=('C' if c_order else 'F'))

@cython.inline
cdef void _edtsq_nd_core_u8_c(uint8_t* labels, float* dest, Py_ssize_t dims, size_t* shape, Py_ssize_t* axes, size_t* strides, float* anis, native_bool black_border) nogil:
  cdef size_t total = 1
  cdef Py_ssize_t i
  cdef bint done, done2
  for i in range(dims): total *= shape[i]
  cdef Py_ssize_t ax0 = axes[0]
  cdef size_t n0 = shape[ax0]
  cdef size_t s0 = strides[ax0]
  # Axis 0: multi-seg using incremental enumeration
  cdef size_t base = 0
  cdef size_t* idx = <size_t*> malloc(dims * sizeof(size_t))
  if idx == NULL:
    return
  for i in range(dims): idx[i] = 0
  # loop over all lines orthogonal to ax0
  while True:
    squared_edt_1d_multi_seg[uint8_t](labels + base, dest + base, <int>n0, <int>s0, anis[ax0], black_border)
    # increment mixed-radix index over dims except ax0
    done = True
    for i in range(dims):
      if i == ax0:
        continue
      idx[i] += 1
      base += strides[i]
      if idx[i] < shape[i]:
        done = False
        break
      base -= idx[i] * strides[i]
      idx[i] = 0
    if done:
      break
  free(idx)
  if not black_border:
    for i in range(total):
      if isinf(dest[i]):
        dest[i] = 3.4028234e+38
  # Remaining axes
  cdef Py_ssize_t aidx
  cdef size_t n, s
  cdef Py_ssize_t ax
  for aidx in range(1, dims):
    ax = axes[aidx]
    n = shape[ax]
    s = strides[ax]
    if n <= 1:
      continue
    base = 0
    idx = <size_t*> malloc(dims * sizeof(size_t))
    if idx == NULL:
      return
    for i in range(dims): idx[i] = 0
    while True:
      squared_edt_1d_parabolic_multi_seg[uint8_t](labels + base, dest + base, <int>n, <int>s, anis[ax], black_border)
      done2 = True
      for i in range(dims):
        if i == ax:
          continue
        idx[i] += 1
        base += strides[i]
        if idx[i] < shape[i]:
          done2 = False
          break
        base -= idx[i] * strides[i]
        idx[i] = 0
      if done2:
        break
    free(idx)
  if not black_border:
    for i in range(total):
      if dest[i] >= 3.4028234e+38:
        dest[i] = INFINITY

@cython.inline
cdef void _edtsq_nd_core_u16_c(uint16_t* labels, float* dest, Py_ssize_t dims, size_t* shape, Py_ssize_t* axes, size_t* strides, float* anis, native_bool black_border) nogil:
  cdef size_t total = 1
  cdef Py_ssize_t i
  cdef bint done, done2
  for i in range(dims): total *= shape[i]
  cdef Py_ssize_t ax0 = axes[0]
  cdef size_t n0 = shape[ax0]
  cdef size_t s0 = strides[ax0]
  cdef size_t base = 0
  cdef size_t* idx = <size_t*> malloc(dims * sizeof(size_t))
  if idx == NULL:
    return
  for i in range(dims): idx[i] = 0
  while True:
    squared_edt_1d_multi_seg[uint16_t](labels + base, dest + base, <int>n0, <int>s0, anis[ax0], black_border)
    done = True
    for i in range(dims):
      if i == ax0:
        continue
      idx[i] += 1
      base += strides[i]
      if idx[i] < shape[i]:
        done = False
        break
      base -= idx[i] * strides[i]
      idx[i] = 0
    if done:
      break
  free(idx)
  if not black_border:
    for i in range(total):
      if isinf(dest[i]):
        dest[i] = 3.4028234e+38
  cdef Py_ssize_t aidx
  cdef size_t n, s
  cdef Py_ssize_t ax
  for aidx in range(1, dims):
    ax = axes[aidx]
    n = shape[ax]
    s = strides[ax]
    if n <= 1:
      continue
    base = 0
    idx = <size_t*> malloc(dims * sizeof(size_t))
    if idx == NULL:
      return
    for i in range(dims): idx[i] = 0
    while True:
      squared_edt_1d_parabolic_multi_seg[uint16_t](labels + base, dest + base, <int>n, <int>s, anis[ax], black_border)
      done2 = True
      for i in range(dims):
        if i == ax:
          continue
        idx[i] += 1
        base += strides[i]
        if idx[i] < shape[i]:
          done2 = False
          break
        base -= idx[i] * strides[i]
        idx[i] = 0
      if done2:
        break
    free(idx)
  if not black_border:
    for i in range(total):
      if dest[i] >= 3.4028234e+38:
        dest[i] = INFINITY

@cython.inline
cdef void _edtsq_nd_core_u32_c(uint32_t* labels, float* dest, Py_ssize_t dims, size_t* shape, Py_ssize_t* axes, size_t* strides, float* anis, native_bool black_border) nogil:
  cdef size_t total = 1
  cdef Py_ssize_t i
  cdef bint done, done2
  for i in range(dims): total *= shape[i]
  cdef Py_ssize_t ax0 = axes[0]
  cdef size_t n0 = shape[ax0]
  cdef size_t s0 = strides[ax0]
  cdef size_t base = 0
  cdef size_t* idx = <size_t*> malloc(dims * sizeof(size_t))
  if idx == NULL:
    return
  for i in range(dims): idx[i] = 0
  while True:
    squared_edt_1d_multi_seg[uint32_t](labels + base, dest + base, <int>n0, <int>s0, anis[ax0], black_border)
    done = True
    for i in range(dims):
      if i == ax0:
        continue
      idx[i] += 1
      base += strides[i]
      if idx[i] < shape[i]:
        done = False
        break
      base -= idx[i] * strides[i]
      idx[i] = 0
    if done:
      break
  free(idx)
  if not black_border:
    for i in range(total):
      if isinf(dest[i]):
        dest[i] = 3.4028234e+38
  cdef Py_ssize_t aidx
  cdef size_t n, s
  cdef Py_ssize_t ax
  for aidx in range(1, dims):
    ax = axes[aidx]
    n = shape[ax]
    s = strides[ax]
    if n <= 1:
      continue
    base = 0
    idx = <size_t*> malloc(dims * sizeof(size_t))
    if idx == NULL:
      return
    for i in range(dims): idx[i] = 0
    while True:
      squared_edt_1d_parabolic_multi_seg[uint32_t](labels + base, dest + base, <int>n, <int>s, anis[ax], black_border)
      done2 = True
      for i in range(dims):
        if i == ax:
          continue
        idx[i] += 1
        base += strides[i]
        if idx[i] < shape[i]:
          done2 = False
          break
        base -= idx[i] * strides[i]
        idx[i] = 0
      if done2:
        break
    free(idx)
  if not black_border:
    for i in range(total):
      if dest[i] >= 3.4028234e+38:
        dest[i] = INFINITY

@cython.inline
cdef void _edtsq_nd_core_u64_c(uint64_t* labels, float* dest, Py_ssize_t dims, size_t* shape, Py_ssize_t* axes, size_t* strides, float* anis, native_bool black_border) nogil:
  cdef size_t total = 1
  cdef Py_ssize_t i
  cdef bint done, done2
  for i in range(dims): total *= shape[i]
  cdef Py_ssize_t ax0 = axes[0]
  cdef size_t n0 = shape[ax0]
  cdef size_t s0 = strides[ax0]
  cdef size_t base = 0
  cdef size_t* idx = <size_t*> malloc(dims * sizeof(size_t))
  if idx == NULL:
    return
  for i in range(dims): idx[i] = 0
  while True:
    squared_edt_1d_multi_seg[uint64_t](labels + base, dest + base, <int>n0, <int>s0, anis[ax0], black_border)
    done = True
    for i in range(dims):
      if i == ax0:
        continue
      idx[i] += 1
      base += strides[i]
      if idx[i] < shape[i]:
        done = False
        break
      base -= idx[i] * strides[i]
      idx[i] = 0
    if done:
      break
  free(idx)
  if not black_border:
    for i in range(total):
      if isinf(dest[i]):
        dest[i] = 3.4028234e+38
  cdef Py_ssize_t aidx
  cdef size_t n, s
  cdef Py_ssize_t ax
  for aidx in range(1, dims):
    ax = axes[aidx]
    n = shape[ax]
    s = strides[ax]
    if n <= 1:
      continue
    base = 0
    idx = <size_t*> malloc(dims * sizeof(size_t))
    if idx == NULL:
      return
    for i in range(dims): idx[i] = 0
    while True:
      squared_edt_1d_parabolic_multi_seg[uint64_t](labels + base, dest + base, <int>n, <int>s, anis[ax], black_border)
      done2 = True
      for i in range(dims):
        if i == ax:
          continue
        idx[i] += 1
        base += strides[i]
        if idx[i] < shape[i]:
          done2 = False
          break
        base -= idx[i] * strides[i]
        idx[i] = 0
      if done2:
        break
    free(idx)
  if not black_border:
    for i in range(total):
      if dest[i] >= 3.4028234e+38:
        dest[i] = INFINITY

@cython.inline
cdef void _edtsq_nd_core_f32_c(float* labels, float* dest, Py_ssize_t dims, size_t* shape, Py_ssize_t* axes, size_t* strides, float* anis, native_bool black_border) nogil:
  cdef size_t total = 1
  cdef Py_ssize_t i
  cdef bint done, done2
  for i in range(dims): total *= shape[i]
  cdef Py_ssize_t ax0 = axes[0]
  cdef size_t n0 = shape[ax0]
  cdef size_t s0 = strides[ax0]
  cdef size_t base = 0
  cdef size_t* idx = <size_t*> malloc(dims * sizeof(size_t))
  if idx == NULL:
    return
  for i in range(dims): idx[i] = 0
  while True:
    squared_edt_1d_multi_seg[float](labels + base, dest + base, <int>n0, <int>s0, anis[ax0], black_border)
    done = True
    for i in range(dims):
      if i == ax0:
        continue
      idx[i] += 1
      base += strides[i]
      if idx[i] < shape[i]:
        done = False
        break
      base -= idx[i] * strides[i]
      idx[i] = 0
    if done:
      break
  free(idx)
  if not black_border:
    for i in range(total):
      if isinf(dest[i]):
        dest[i] = 3.4028234e+38
  cdef Py_ssize_t aidx
  cdef size_t n, s
  cdef Py_ssize_t ax
  for aidx in range(1, dims):
    ax = axes[aidx]
    n = shape[ax]
    s = strides[ax]
    if n <= 1:
      continue
    base = 0
    idx = <size_t*> malloc(dims * sizeof(size_t))
    if idx == NULL:
      return
    for i in range(dims): idx[i] = 0
    while True:
      squared_edt_1d_parabolic_multi_seg[float](labels + base, dest + base, <int>n, <int>s, anis[ax], black_border)
      done2 = True
      for i in range(dims):
        if i == ax:
          continue
        idx[i] += 1
        base += strides[i]
        if idx[i] < shape[i]:
          done2 = False
          break
        base -= idx[i] * strides[i]
        idx[i] = 0
      if done2:
        break
    free(idx)
  if not black_border:
    for i in range(total):
      if dest[i] >= 3.4028234e+38:
        dest[i] = INFINITY

@cython.inline
cdef void _edtsq_nd_core_f64_c(double* labels, float* dest, Py_ssize_t dims, size_t* shape, Py_ssize_t* axes, size_t* strides, float* anis, native_bool black_border) nogil:
  cdef size_t total = 1
  cdef Py_ssize_t i
  cdef bint done, done2
  for i in range(dims): total *= shape[i]
  cdef Py_ssize_t ax0 = axes[0]
  cdef size_t n0 = shape[ax0]
  cdef size_t s0 = strides[ax0]
  cdef size_t base = 0
  cdef size_t* idx = <size_t*> malloc(dims * sizeof(size_t))
  if idx == NULL:
    return
  for i in range(dims): idx[i] = 0
  while True:
    squared_edt_1d_multi_seg[double](labels + base, dest + base, <int>n0, <int>s0, anis[ax0], black_border)
    done = True
    for i in range(dims):
      if i == ax0:
        continue
      idx[i] += 1
      base += strides[i]
      if idx[i] < shape[i]:
        done = False
        break
      base -= idx[i] * strides[i]
      idx[i] = 0
    if done:
      break
  free(idx)
  if not black_border:
    for i in range(total):
      if isinf(dest[i]):
        dest[i] = 3.4028234e+38
  cdef Py_ssize_t aidx
  cdef size_t n, s
  cdef Py_ssize_t ax
  for aidx in range(1, dims):
    ax = axes[aidx]
    n = shape[ax]
    s = strides[ax]
    if n <= 1:
      continue
    base = 0
    idx = <size_t*> malloc(dims * sizeof(size_t))
    if idx == NULL:
      return
    for i in range(dims): idx[i] = 0
    while True:
      squared_edt_1d_parabolic_multi_seg[double](labels + base, dest + base, <int>n, <int>s, anis[ax], black_border)
      done2 = True
      for i in range(dims):
        if i == ax:
          continue
        idx[i] += 1
        base += strides[i]
        if idx[i] < shape[i]:
          done2 = False
          break
        base -= idx[i] * strides[i]
        idx[i] = 0
      if done2:
        break
    free(idx)
  if not black_border:
    for i in range(total):
      if dest[i] >= 3.4028234e+38:
        dest[i] = INFINITY

@cython.inline
cdef void _edtsq_nd_core_bool_c(native_bool* labels, float* dest, Py_ssize_t dims, size_t* shape, Py_ssize_t* axes, size_t* strides, float* anis, native_bool black_border) nogil:
  cdef size_t total = 1
  cdef Py_ssize_t i
  cdef bint done, done2
  for i in range(dims): total *= shape[i]
  cdef Py_ssize_t ax0 = axes[0]
  cdef size_t n0 = shape[ax0]
  cdef size_t s0 = strides[ax0]
  
  cdef size_t* idx = <size_t*> malloc(dims * sizeof(size_t))
  if idx == NULL:
    return
  for i in range(dims): idx[i] = 0
  cdef size_t base = 0
  while True:
    squared_edt_1d_multi_seg[native_bool](labels + base, dest + base, <int>n0, <int>s0, anis[ax0], black_border)
    done = True
    for i in range(dims):
      if i == ax0:
        continue
      idx[i] += 1
      base += strides[i]
      if idx[i] < shape[i]:
        done = False
        break
      base -= idx[i] * strides[i]
      idx[i] = 0
    if done:
      break
  if not black_border:
    for i in range(total):
      if isinf(dest[i]):
        dest[i] = 3.4028234e+38
  cdef Py_ssize_t aidx
  cdef size_t n, s
  cdef Py_ssize_t ax
  for aidx in range(1, dims):
    ax = axes[aidx]
    n = shape[ax]
    s = strides[ax]
    if n <= 1:
      continue
    base = 0
    for i in range(dims): idx[i] = 0
    while True:
      squared_edt_1d_parabolic_multi_seg[native_bool](labels + base, dest + base, <int>n, <int>s, anis[ax], black_border)
      done2 = True
      for i in range(dims):
        if i == ax:
          continue
        idx[i] += 1
        base += strides[i]
        if idx[i] < shape[i]:
          done2 = False
          break
        base -= idx[i] * strides[i]
        idx[i] = 0
      if done2:
        break
  free(idx)
  if not black_border:
    for i in range(total):
      if dest[i] >= 3.4028234e+38:
        dest[i] = INFINITY

@cython.binding(True)
def edt_nd(
    data, anisotropy=None, native_bool black_border=False, 
    int parallel=1, voxel_graph=None, order=None,
  ):
  res = edtsq_nd(data, anisotropy, black_border, parallel, voxel_graph, order)
  return np.sqrt(res, res)


def edt3d(
    data, anisotropy=(1.0, 1.0, 1.0), 
    native_bool black_border=False,
    parallel=1, voxel_graph=None
  ):
  result = edt3dsq(data, anisotropy, black_border, parallel, voxel_graph)
  return np.sqrt(result, result)

def edt3dsq(
    data, anisotropy=(1.0, 1.0, 1.0), 
    native_bool black_border=False,
    int parallel=1, voxel_graph=None
  ):
  if voxel_graph is not None:
    return __edt3dsq_voxel_graph(data, voxel_graph, anisotropy, black_border)
  return __edt3dsq(data, anisotropy, black_border, parallel)

def __edt3dsq(
    data, anisotropy=(1.0, 1.0, 1.0), 
    native_bool black_border=False,
    int parallel=1
  ):
  cdef uint8_t[:,:,:] arr_memview8
  cdef uint16_t[:,:,:] arr_memview16
  cdef uint32_t[:,:,:] arr_memview32
  cdef uint64_t[:,:,:] arr_memview64
  cdef float[:,:,:] arr_memviewfloat
  cdef double[:,:,:] arr_memviewdouble

  cdef size_t sx = data.shape[2]
  cdef size_t sy = data.shape[1]
  cdef size_t sz = data.shape[0]
  cdef float ax = anisotropy[2]
  cdef float ay = anisotropy[1]
  cdef float az = anisotropy[0]

  order = 'C'
  if data.flags.f_contiguous:
    sx, sy, sz = sz, sy, sx
    ax = anisotropy[0]
    ay = anisotropy[1]
    az = anisotropy[2]
    order = 'F'

  cdef size_t voxels = sx * sy * sz
  cdef np.ndarray[float, ndim=1] output = np.zeros( (voxels,), dtype=np.float32 )
  cdef float[:] outputview = output

  if data.dtype in (np.uint8, np.int8):
    arr_memview8 = data.astype(np.uint8)
    _edt3dsq_with_features[uint8_t, size_t](
      <uint8_t*>&arr_memview8[0,0,0],
      sx, sy, sz,
      ax, ay, az,
      black_border, parallel,
      <float*>&outputview[0], NULL
    )
  elif data.dtype in (np.uint16, np.int16):
    arr_memview16 = data.astype(np.uint16)
    _edt3dsq_with_features[uint16_t, size_t](
      <uint16_t*>&arr_memview16[0,0,0],
      sx, sy, sz,
      ax, ay, az,
      black_border, parallel,
      <float*>&outputview[0], NULL
    )
  elif data.dtype in (np.uint32, np.int32):
    arr_memview32 = data.astype(np.uint32)
    _edt3dsq_with_features[uint32_t, size_t](
      <uint32_t*>&arr_memview32[0,0,0],
      sx, sy, sz,
      ax, ay, az,
      black_border, parallel,
      <float*>&outputview[0], NULL
    )
  elif data.dtype in (np.uint64, np.int64):
    arr_memview64 = data.astype(np.uint64)
    _edt3dsq_with_features[uint64_t, size_t](
      <uint64_t*>&arr_memview64[0,0,0],
      sx, sy, sz,
      ax, ay, az,
      black_border, parallel,
      <float*>&outputview[0], NULL
    )
  elif data.dtype == np.float32:
    arr_memviewfloat = data
    _edt3dsq_with_features[float, size_t](
      <float*>&arr_memviewfloat[0,0,0],
      sx, sy, sz,
      ax, ay, az,
      black_border, parallel,
      <float*>&outputview[0], NULL
    )
  elif data.dtype == np.float64:
    arr_memviewdouble = data
    _edt3dsq_with_features[double, size_t](
      <double*>&arr_memviewdouble[0,0,0],
      sx, sy, sz,
      ax, ay, az,
      black_border, parallel,
      <float*>&outputview[0], NULL
    )
  elif data.dtype == bool:
    arr_memview8 = data.view(np.uint8)
    _edt3dsq_with_features[native_bool, size_t](
      <native_bool*>&arr_memview8[0,0,0],
      sx, sy, sz,
      ax, ay, az,
      black_border, parallel,
      <float*>&outputview[0], NULL
    )

  return output.reshape( data.shape, order=order)

def __edt3dsq_voxel_graph(
    data, voxel_graph, 
    anisotropy=(1.0, 1.0, 1.0), 
    native_bool black_border=False,
  ):
  cdef uint8_t[:,:,:] arr_memview8
  cdef uint16_t[:,:,:] arr_memview16
  cdef uint32_t[:,:,:] arr_memview32
  cdef uint64_t[:,:,:] arr_memview64
  cdef float[:,:,:] arr_memviewfloat
  cdef double[:,:,:] arr_memviewdouble

  cdef uint8_t[:,:,:] graph_memview8
  if voxel_graph.dtype in (np.uint8, np.int8):
    graph_memview8 = voxel_graph.view(np.uint8)
  else:
    graph_memview8 = voxel_graph.astype(np.uint8) # we only need first 6 bits

  cdef size_t sx = data.shape[2]
  cdef size_t sy = data.shape[1]
  cdef size_t sz = data.shape[0]
  cdef float ax = anisotropy[2]
  cdef float ay = anisotropy[1]
  cdef float az = anisotropy[0]
  order = 'C'

  if data.flags.f_contiguous:
    sx, sy, sz = sz, sy, sx
    ax = anisotropy[0]
    ay = anisotropy[1]
    az = anisotropy[2]
    order = 'F'

  cdef size_t voxels = sx * sy * sz
  cdef np.ndarray[float, ndim=1] output = np.zeros( (voxels,), dtype=np.float32 )
  cdef float[:] outputview = output

  if data.dtype in (np.uint8, np.int8):
    arr_memview8 = data.astype(np.uint8)
    _edt3dsq_voxel_graph[uint8_t,uint8_t](
      <uint8_t*>&arr_memview8[0,0,0],
      <uint8_t*>&graph_memview8[0,0,0],
      sx, sy, sz,
      ax, ay, az,
      black_border,
      <float*>&outputview[0]
    )
  elif data.dtype in (np.uint16, np.int16):
    arr_memview16 = data.astype(np.uint16)
    _edt3dsq_voxel_graph[uint16_t,uint8_t](
      <uint16_t*>&arr_memview16[0,0,0], 
      <uint8_t*>&graph_memview8[0,0,0],
      sx, sy, sz,
      ax, ay, az,
      black_border,
      <float*>&outputview[0]
    )
  elif data.dtype in (np.uint32, np.int32):
    arr_memview32 = data.astype(np.uint32)
    _edt3dsq_voxel_graph[uint32_t,uint8_t](
      <uint32_t*>&arr_memview32[0,0,0],
      <uint8_t*>&graph_memview8[0,0,0],
      sx, sy, sz,
      ax, ay, az,
      black_border,
      <float*>&outputview[0]
    )
  elif data.dtype in (np.uint64, np.int64):
    arr_memview64 = data.astype(np.uint64)
    _edt3dsq_voxel_graph[uint64_t,uint8_t](
      <uint64_t*>&arr_memview64[0,0,0],
      <uint8_t*>&graph_memview8[0,0,0],
      sx, sy, sz,
      ax, ay, az,
      black_border,
      <float*>&outputview[0]
    )
  elif data.dtype == np.float32:
    arr_memviewfloat = data
    _edt3dsq_voxel_graph[float,uint8_t](
      <float*>&arr_memviewfloat[0,0,0],
      <uint8_t*>&graph_memview8[0,0,0],
      sx, sy, sz,
      ax, ay, az,
      black_border,
      <float*>&outputview[0]
    )
  elif data.dtype == np.float64:
    arr_memviewdouble = data
    _edt3dsq_voxel_graph[double,uint8_t](
      <double*>&arr_memviewdouble[0,0,0],
      <uint8_t*>&graph_memview8[0,0,0],
      sx, sy, sz,
      ax, ay, az,
      black_border,
      <float*>&outputview[0]
    )
  elif data.dtype == bool:
    arr_memview8 = data.view(np.uint8)
    _edt3dsq_voxel_graph[native_bool,uint8_t](
      <native_bool*>&arr_memview8[0,0,0],
      <uint8_t*>&graph_memview8[0,0,0],
      sx, sy, sz,
      ax, ay, az,
      black_border,
      <float*>&outputview[0]
    )

  return output.reshape(data.shape, order=order)


## These below functions are concerned with fast rendering
## of a densely labeled image into a series of binary images.

# from https://github.com/seung-lab/fastremap/blob/master/fastremap.pyx
def reshape(arr, shape, order=None):
  """
  If the array is contiguous, attempt an in place reshape
  rather than potentially making a copy.
  Required:
    arr: The input numpy array.
    shape: The desired shape (must be the same size as arr)
  Optional: 
    order: 'C', 'F', or None (determine automatically)
  Returns: reshaped array
  """
  if order is None:
    if arr.flags['F_CONTIGUOUS']:
      order = 'F'
    elif arr.flags['C_CONTIGUOUS']:
      order = 'C'
    else:
      return arr.reshape(shape)

  cdef int nbytes = np.dtype(arr.dtype).itemsize

  if order == 'C':
    strides = [ reduce(operator.mul, shape[i:]) * nbytes for i in range(1, len(shape)) ]
    strides += [ nbytes ]
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
  else:
    strides = [ reduce(operator.mul, shape[:i]) * nbytes for i in range(1, len(shape)) ]
    strides = [ nbytes ] + strides
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

# from https://github.com/seung-lab/connected-components-3d/blob/master/cc3d.pyx
def runs(labels):
  """
  runs(labels)

  Returns a dictionary describing where each label is located.
  Use this data in conjunction with render and erase.
  """
  return _runs(reshape(labels, (labels.size,)))

def _runs(
    np.ndarray[NUMBER, ndim=1, cast=True] labels
  ):
  return extract_runs(<NUMBER*>&labels[0], <size_t>labels.size)

def draw(
  label, 
  vector[cpp_pair[size_t, size_t]] runs,
  image
):
  """
  draw(label, runs, image)

  Draws label onto the provided image according to 
  runs.
  """
  return _draw(label, runs, reshape(image, (image.size,)))

def _draw( 
  label, 
  vector[cpp_pair[size_t, size_t]] runs,
  np.ndarray[NUMBER, ndim=1, cast=True] image
):
  set_run_voxels(<NUMBER>label, runs, <NUMBER*>&image[0], <size_t>image.size)
  return image

def transfer(
  vector[cpp_pair[size_t, size_t]] runs,
  src, dest
):
  """
  transfer(runs, src, dest)

  Transfers labels from source to destination image 
  according to runs.
  """
  return _transfer(runs, reshape(src, (src.size,)), reshape(dest, (dest.size,)))

def _transfer( 
  vector[cpp_pair[size_t, size_t]] runs,
  np.ndarray[floating, ndim=1, cast=True] src,
  np.ndarray[floating, ndim=1, cast=True] dest,
):
  assert src.size == dest.size
  transfer_run_voxels(runs, <floating*>&src[0], <floating*>&dest[0], src.size)
  return dest

def erase( 
  vector[cpp_pair[size_t, size_t]] runs, 
  image
):
  """
  erase(runs, image)

  Erases (sets to 0) part of the provided image according to 
  runs.
  """
  return draw(0, runs, image)

@cython.binding(True)
def each(labels, dt, in_place=False):
  """
  Returns an iterator that extracts each label's distance transform.
  labels is the original labels the distance transform was calculated from.
  dt is the distance transform.

  in_place: much faster but the resulting image will be read-only

  Example:
  for label, img in cc3d.each(labels, dt, in_place=False):
    process(img)

  Returns: iterator
  """
  all_runs = runs(labels)
  order = 'F' if labels.flags.f_contiguous else 'C'
  dtype = np.float32

  class ImageIterator():
    def __len__(self):
      return len(all_runs) - int(0 in all_runs)
    def __iter__(self):
      for key, rns in all_runs.items():
        if key == 0:
          continue
        img = np.zeros(labels.shape, dtype=dtype, order=order)
        transfer(rns, dt, img)
        yield (key, img)

  class InPlaceImageIterator(ImageIterator):
    def __iter__(self):
      img = np.zeros(labels.shape, dtype=dtype, order=order)
      for key, rns in all_runs.items():
        if key == 0:
          continue
        transfer(rns, dt, img)
        img.setflags(write=0)
        yield (key, img)
        img.setflags(write=1)
        erase(rns, img)

  if in_place:
    return InPlaceImageIterator()
  return ImageIterator()

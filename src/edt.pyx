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

    cdef float* _edt3dsq_with_features[T, OUTIDX](
        T* input, size_t sx, size_t sy, size_t sz,
        float wx, float wy, float wz,
        native_bool black_border, int parallel,
        float* output_dt, OUTIDX* output_feat
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
  cdef np.ndarray pos
  cdef np.ndarray mids
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
    # Fast NumPy-based implementation
    out2 = np.zeros_like(arr)
    
    # Find all seed positions and their labels
    seed_positions = np.argwhere(arr != 0)
    if len(seed_positions) == 0:
      return out2
    
    seed_labels = arr[seed_positions[:, 0], seed_positions[:, 1]]
    
    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
    
    # For each seed, calculate distances to all pixels
    min_distances = np.full((arr.shape[0], arr.shape[1]), np.inf)
    nearest_labels = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint32)
    
    for i, (y, x) in enumerate(seed_positions):
      distances = (y_coords - y)**2 + (x_coords - x)**2
      mask = distances < min_distances
      min_distances[mask] = distances[mask]
      nearest_labels[mask] = seed_labels[i]
    
    # Set seed positions to their original values
    out2[seed_positions[:, 0], seed_positions[:, 1]] = seed_labels
    
    # Set non-seed positions to nearest label
    seed_mask = arr != 0
    out2[~seed_mask] = nearest_labels[~seed_mask]
    
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



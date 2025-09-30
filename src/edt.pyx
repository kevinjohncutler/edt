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
  edt_nd, edtsq_nd
  feature_transform_nd, expand_labels_nd

Legacy dimension-specific APIs are available through ``edt.original`` when the
reference repository has been cloned and built locally.

License: GNU 3.0

Author: William Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institute
Date: July 2018 - December 2023
"""
import importlib.util
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
import os
import pathlib
import time
import sys
import types


_nd_profile_last = None


class _OriginalModuleProxy(types.ModuleType):
  """Lazy loader for the legacy dimension-specific EDT extension."""

  def __init__(self):
    super().__init__('edt.original')
    self._module = None
    self._load_error = None

  def _candidate_roots(self):
    env_path = os.environ.get('EDT_ORIGINAL_PATH')
    if env_path:
      path = pathlib.Path(env_path).expanduser()
      if path.exists():
        yield path
    base = pathlib.Path(__file__).resolve()
    seen = set()
    for parent in (base.parent,) + tuple(base.parents):
      candidate = (parent / 'original_repo').resolve()
      if candidate.exists() and candidate not in seen:
        seen.add(candidate)
        yield candidate

  def _iter_extension_candidates(self, roots):
    patterns = ('**/edt*.so', '**/edt*.pyd', '**/edt*.dll')
    for root in roots:
      for pattern in patterns:
        for path in sorted(root.glob(pattern), reverse=True):
          if path.is_file():
            yield path

  def _load(self):
    if self._module is not None:
      return self._module
    if self._load_error is not None:
      raise self._load_error

    try:
      module = importlib.import_module('edt_original')
      module.__name__ = 'edt.original'
      module.available = lambda: True
      sys.modules['edt.original'] = module
      self._module = module
      self._load_error = None
      return module
    except Exception as exc:
      self._load_error = exc

    # fall back to disk lookup (for developers with external clone)
    roots = list(self._candidate_roots())
    candidates = list(self._iter_extension_candidates(roots))
    errors = []

    for candidate in candidates:
      try:
        spec = importlib.util.spec_from_file_location('edt_original_native', candidate)
        if spec is None or spec.loader is None:
          raise ImportError(f'Unable to create loader for {candidate}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.__name__ = 'edt.original'
        module.available = lambda: True
        sys.modules['edt.original'] = module
        self._module = module
        self._load_error = None
        return module
      except Exception as exc:
        errors.append((candidate, exc))

    if errors:
      messages = ', '.join(f"{path.name}: {exc}" for path, exc in errors)
      err = ImportError(f'Failed to load original EDT extension ({messages})')
      self._load_error = err
      raise err

    if roots:
      hint = roots[0]
      err = ImportError(
        f'Original EDT extension not built in {hint}. '
        'Run `pip install -e .` there or build the packaged edt.original._native.'
      )
      self._load_error = err
      raise err
    err = ImportError(
      "Original EDT implementation not available. Install the package with the"
      " bundled edt.original module or provide a build in ./original_repo."
    )
    self._load_error = err
    raise err

  def __getattr__(self, name):
    module = self._load()
    return getattr(module, name)

  def __dir__(self):
    try:
      module = self._load()
    except Exception:
      return sorted(set(super().__dir__()) | {'available'})
    return sorted(set(super().__dir__()) | set(dir(module)) | {'available'})

  def available(self):
    """Return True if the legacy module could be loaded."""
    if self._module is not None:
      return True
    try:
      self._load()
    except Exception:
      return False
    return True


original = _OriginalModuleProxy()

@cython.binding(True)
def nd_tuning(tile=None, prefetch_step=None, chunks_per_thread=None):
  """Tune ND internals: lines tile size, prefetch lookahead, and chunking.

  Pass ints or None. None leaves the underlying setting unchanged.
  """
  cdef size_t t = 0
  cdef size_t p = 0
  cdef size_t c = 0
  if tile is not None:
    t = <size_t>int(tile)
  if prefetch_step is not None:
    p = <size_t>int(prefetch_step)
  if chunks_per_thread is not None:
    c = <size_t>int(chunks_per_thread)
  with nogil:
    nd_set_tuning(t, p, c)

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
cdef extern from "edt.hpp" namespace "pyedt":
    cdef void squared_edt_1d_multi_seg[T](
        T *labels,
        float *dest,
        int n,
        int stride,
        float anisotropy,
        native_bool black_border
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
        INDEX* feat,
        int parallel
    ) nogil

    # Tuning hook for ND threading/tiling
    cdef void nd_set_tuning(size_t tile, size_t prefetch_step, size_t chunks_per_thread) nogil

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

cdef extern from "edt_voxel_graph.hpp" namespace "pyedt":
  cdef mapcpp[T, vector[cpp_pair[size_t, size_t]]] extract_runs[T](
    T* labels, size_t voxels
  )
  cdef void set_run_voxels[T](
    T key,
    vector[cpp_pair[size_t, size_t]] all_runs,
    T* labels, size_t voxels
  ) except +
  cdef void transfer_run_voxels[T](
    vector[cpp_pair[size_t, size_t]] all_runs,
    T* src, T* dest,
    size_t voxels
  ) except +


def _adaptive_thread_limit_nd(parallel, shape):
  """General ND limiter that matches 2D/3D heuristics and scales for higher dims."""
  try:
    if not bool(int(os.environ.get('EDT_ADAPTIVE_THREADS', '1'))):
      return parallel
  except Exception:
    pass

  dims = len(shape)
  if dims <= 1:
    return parallel

  # Match existing tuned behaviour for 2D and 3D so benchmarks line up.
  if dims == 2:
    total = shape[0] * shape[1]
    if total <= 16384:  # <= 128x128
      return min(parallel, 4)
    if total <= 262144:  # <= 512x512
      return min(parallel, 8)
    if parallel >= 16:
      return min(parallel, 12)
    return parallel

  if dims == 3:
    max_dim = max(shape)
    if max_dim <= 32:
      return min(parallel, 4)
    if max_dim <= 48:
      return min(parallel, 8)
    if max_dim <= 64:
      return min(parallel, 12)
    return parallel

  # Higher-D fallback: ensure every worker processes enough lines and voxels.
  total = 1
  for extent in shape:
    total *= extent
  longest = max(shape)
  lines = max(1, total // longest)

  # Require a minimum amount of work per thread to avoid oversubscription.
  min_lines_per_thread = 64
  min_voxels_per_thread = 32768

  cap_lines = max(1, lines // min_lines_per_thread)
  cap_voxels = max(1, total // min_voxels_per_thread)
  cap = max(cap_lines, cap_voxels)
  return max(1, min(parallel, cap))

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
  cdef uint32_t* fprev_u32
  cdef size_t* fprev_sz
  cdef np.ndarray[np.uint32_t, ndim=1] out_flat
  cdef np.uint32_t* outp
  cdef np.uint32_t* labp2
  cdef size_t idx2
  cdef Py_ssize_t total_voxels
  cdef Py_ssize_t thresh8
  cdef Py_ssize_t thresh12
  cdef Py_ssize_t thresh16
  cdef Py_ssize_t max_dim
  cdef Py_ssize_t total_elems
  cdef int target

  arr = np.require(data, dtype=np.uint32, requirements='C')
  dims = arr.ndim

  cdef bint adapt_threads_enabled = True
  try:
    adapt_threads_enabled = bool(int(os.environ.get('EDT_ADAPTIVE_THREADS', '1')))
  except Exception:
    adapt_threads_enabled = True

  cdef int cpu_cap = 1
  try:
    cpu_cap = multiprocessing.cpu_count()
  except Exception:
    cpu_cap = max(1, parallel)

  if parallel <= 0:
    parallel = cpu_cap
  else:
    parallel = max(1, min(parallel, cpu_cap))

  if adapt_threads_enabled and dims == 3:
    total_voxels = arr.size
    thresh8 = 128 * 128 * 128
    thresh12 = 192 * 192 * 192
    thresh16 = 256 * 256 * 256
    target = 0
    if total_voxels >= thresh16:
      target = min(cpu_cap, 16)
    elif total_voxels >= thresh12:
      target = min(cpu_cap, 12)
    elif total_voxels >= thresh8:
      target = min(cpu_cap, 8)
    if target > parallel:
      parallel = target
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
  fprev_u32 = NULL
  fprev_sz = NULL
  if return_features:
    if use_u32_feat:
      feat_prev = np.empty((total,), dtype=np.uint32)
      fprev_u32 = <uint32_t*> np.PyArray_DATA(feat_prev)
    else:
      feat_prev = np.empty((total,), dtype=np.uintp)
      fprev_sz = <size_t*> np.PyArray_DATA(feat_prev)
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
          _nd_expand_parabolic_bases[uint32_t](distp, bases, lines, n0, s0, canis[paxes[a]], bb, fprev_u32, parallel)
        else:
          _nd_expand_parabolic_bases[size_t](distp, bases, lines, n0, s0, canis[paxes[a]], bb, fprev_sz, parallel)
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
def expand_labels(
    data, anisotropy=None, black_border=False,
    int parallel=1, voxel_graph=None):
  """Compatibility wrapper that forwards to :func:`expand_labels_nd`."""
  return expand_labels_nd(data, anisotropy, black_border, parallel, voxel_graph, False)


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


@cython.binding(True)
def feature_transform(data, anisotropy=None, black_border=False,
                      int parallel=1, voxel_graph=None,
                      return_distances=False, features_dtype='auto'):
  """Compatibility wrapper that forwards to :func:`feature_transform_nd`."""
  return feature_transform_nd(data, anisotropy, black_border, parallel, voxel_graph,
                              return_distances, features_dtype)
def __getattr__(name):
  """Forward unknown attributes to the packaged legacy module."""
  try:
    attr = getattr(original, name)
  except AttributeError as exc:
    raise AttributeError(name) from exc
  setattr(sys.modules[__name__], name, attr)
  return attr


@cython.binding(True)
def edt_nd(
    data, anisotropy=None, native_bool black_border=False, 
    int parallel=1, voxel_graph=None, order=None,
  ):
  res = edtsq_nd(data, anisotropy, black_border, parallel, voxel_graph, order)
  return np.sqrt(res, res)

@cython.binding(True)
def edtsq_nd(
  data, anisotropy=None, native_bool black_border=False,
  int parallel=1, voxel_graph=None, order=None,
):
  """General ND squared EDT using a dimension-agnostic pass schedule."""
  global _nd_profile_last
  _nd_profile_last = None

  cdef bint profile_enabled = False
  cdef double t_total_start = 0.0
  cdef dict profile_data = {}
  cdef dict profile_sections = {}
  cdef list profile_axes = []
  cdef Py_ssize_t total_voxels
  cdef Py_ssize_t thresh8
  cdef Py_ssize_t thresh12
  cdef Py_ssize_t thresh16
  cdef int target

  profile_env = os.environ.get('EDT_ND_PROFILE')
  if profile_env:
    try:
      profile_enabled = bool(int(profile_env))
    except Exception:
      profile_enabled = True
  if profile_enabled:
    t_total_start = time.perf_counter()

  requested_parallel = parallel

  if isinstance(data, list):
    data = np.array(data)
  arr = np.asarray(data)
  if arr.size == 0:
    if profile_enabled:
      now = time.perf_counter()
      _nd_profile_last = {
        'shape': tuple(arr.shape),
        'dims': arr.ndim,
        'dtype': str(arr.dtype),
        'requested_parallel': requested_parallel,
        'parallel_used': 0,
        'sections': {'prep': now - t_total_start, 'total': now - t_total_start},
        'axes': []
      }
    return np.zeros_like(arr, dtype=np.float32)
  if not arr.flags.c_contiguous and not arr.flags.f_contiguous:
    arr = np.ascontiguousarray(arr)

  dims = arr.ndim
  if anisotropy is None:
    anis = (1.0,) * dims
  else:
    anis = tuple(anisotropy) if hasattr(anisotropy, '__len__') else (float(anisotropy),) * dims
    if len(anis) != dims:
      raise ValueError('anisotropy length must match data.ndim')

  if profile_enabled:
    profile_data = {
      'shape': tuple(arr.shape),
      'dims': dims,
      'dtype': str(arr.dtype),
      'requested_parallel': requested_parallel,
      'parallel_used': None,
      'sections': {},
      'axes': profile_axes,
    }
    profile_sections = profile_data['sections']
    profile_sections['prep'] = time.perf_counter() - t_total_start

  cdef bint adapt_threads_enabled = True
  try:
    adapt_threads_enabled = bool(int(os.environ.get('EDT_ADAPTIVE_THREADS', '1')))
  except Exception:
    adapt_threads_enabled = True

  cdef int cpu_cap = 1
  try:
    cpu_cap = multiprocessing.cpu_count()
  except Exception:
    cpu_cap = max(1, parallel)

  cdef int p = parallel
  if p <= 0:
    p = cpu_cap
  else:
    p = max(1, min(p, cpu_cap))

  if adapt_threads_enabled and dims == 3:
    total_voxels = arr.size
    thresh8 = 128 * 128 * 128
    thresh12 = 192 * 192 * 192
    thresh16 = 256 * 256 * 256
    target = 0
    if total_voxels >= thresh16:
      target = min(cpu_cap, 16)
    elif total_voxels >= thresh12:
      target = min(cpu_cap, 12)
    elif total_voxels >= thresh8:
      target = min(cpu_cap, 8)
    if target > p:
      p = target

  parallel = _adaptive_thread_limit_nd(p, arr.shape)

  if profile_enabled:
    profile_data['parallel_used'] = parallel

  if voxel_graph is not None:
    raise TypeError('voxel_graph is only supported by 2D/3D specialized APIs')

  cdef Py_ssize_t nd = dims
  cdef size_t* cshape = <size_t*> malloc(nd * sizeof(size_t))
  cdef size_t* cstrides = <size_t*> malloc(nd * sizeof(size_t))
  if cshape == NULL or cstrides == NULL:
    if cshape != NULL: free(cshape)
    if cstrides != NULL: free(cstrides)
    raise MemoryError('Allocation failure')

  cdef size_t total = 1
  cdef Py_ssize_t i
  for i in range(nd):
    cshape[i] = <size_t>arr.shape[i]
  if arr.flags.c_contiguous:
    cstrides[nd-1] = 1
    for i in range(nd-2, -1, -1):
      cstrides[i] = cstrides[i+1] * cshape[i+1]
  else:
    cstrides[0] = 1
    for i in range(1, nd):
      cstrides[i] = cstrides[i-1] * cshape[i-1]
  for i in range(nd):
    total *= cshape[i]

  cdef size_t tune_tile = 0
  cdef size_t tune_prefetch = 0
  cdef size_t tune_chunks = 0
  if parallel > 1:
    if dims == 3:
      max_dim = max(arr.shape)
      if max_dim >= 256:
        tune_tile = 64; tune_prefetch = 4; tune_chunks = <size_t>max(1, parallel // 2)
      elif max_dim >= 128:
        tune_tile = 32; tune_prefetch = 2; tune_chunks = <size_t>max(1, parallel // 2)
      elif max_dim >= 64:
        tune_tile = 16; tune_prefetch = 2; tune_chunks = <size_t>max(1, parallel // 2)
      else:
        tune_tile = 8; tune_prefetch = 2; tune_chunks = <size_t>max(1, parallel)
    elif dims == 2:
      total_elems = arr.shape[0] * arr.shape[1]
      if total_elems >= 1024 * 1024:
        tune_tile = 64; tune_prefetch = 3; tune_chunks = <size_t>max(1, parallel // 2)
      elif total_elems >= 256 * 256:
        tune_tile = 32; tune_prefetch = 2; tune_chunks = <size_t>max(1, parallel // 2)
      elif total_elems >= 128 * 128:
        tune_tile = 16; tune_prefetch = 2; tune_chunks = <size_t>max(1, parallel // 2)
      else:
        tune_tile = 8; tune_prefetch = 2; tune_chunks = <size_t>max(1, parallel)
    else:
      tune_tile = 16 if dims <= 4 else 8
      tune_prefetch = 2
      tune_chunks = <size_t>max(1, parallel // 2)
    with nogil:
      nd_set_tuning(tune_tile, tune_prefetch, tune_chunks)

  cdef np.ndarray out = np.zeros((<Py_ssize_t> total,), dtype=np.float32)
  cdef float* outp = <float*> np.PyArray_DATA(out)
  cdef void* datap = <void*> np.PyArray_DATA(arr.ravel(order='K'))

  cdef double t_tmp
  cdef double t_axis
  cdef double multi_elapsed = 0.0
  cdef double parabolic_elapsed = 0.0

  if nd == 1:
    if profile_enabled:
      t_tmp = time.perf_counter()
    if arr.dtype in (np.uint8, np.int8):
      squared_edt_1d_multi_seg[uint8_t](<uint8_t*>datap, outp, <int>cshape[0], 1, <float>anis[0], black_border)
    elif arr.dtype in (np.uint16, np.int16):
      squared_edt_1d_multi_seg[uint16_t](<uint16_t*>datap, outp, <int>cshape[0], 1, <float>anis[0], black_border)
    elif arr.dtype in (np.uint32, np.int32):
      squared_edt_1d_multi_seg[uint32_t](<uint32_t*>datap, outp, <int>cshape[0], 1, <float>anis[0], black_border)
    elif arr.dtype in (np.uint64, np.int64):
      squared_edt_1d_multi_seg[uint64_t](<uint64_t*>datap, outp, <int>cshape[0], 1, <float>anis[0], black_border)
    elif arr.dtype == np.float32:
      squared_edt_1d_multi_seg[float](<float*>datap, outp, <int>cshape[0], 1, <float>anis[0], black_border)
    elif arr.dtype == np.float64:
      squared_edt_1d_multi_seg[double](<double*>datap, outp, <int>cshape[0], 1, <float>anis[0], black_border)
    elif arr.dtype == bool:
      squared_edt_1d_multi_seg[native_bool](<native_bool*>datap, outp, <int>cshape[0], 1, <float>anis[0], black_border)
    if profile_enabled:
      multi_elapsed = time.perf_counter() - t_tmp
      profile_axes.append({'axis': 0, 'kind': 'multi', 'time': multi_elapsed})
      profile_sections['multi_pass'] = multi_elapsed
      profile_sections['parabolic_pass'] = 0.0
      profile_sections['total'] = time.perf_counter() - t_total_start
      _nd_profile_last = profile_data
    free(cshape); free(cstrides)
    order_ch2 = 'C' if arr.flags.c_contiguous else 'F'
    return np.reshape(out, arr.shape, order=order_ch2)

  cdef Py_ssize_t* axes = <Py_ssize_t*> malloc(nd * sizeof(Py_ssize_t))
  if axes == NULL:
    free(cshape); free(cstrides)
    raise MemoryError('Allocation failure')
  for i in range(nd):
    axes[i] = i
  cdef Py_ssize_t a, j, key
  for a in range(1, nd):
    key = axes[a]
    j = a - 1
    while j >= 0 and (
      cstrides[axes[j]] > cstrides[key]
      or (cstrides[axes[j]] == cstrides[key] and cshape[axes[j]] < cshape[key])
    ):
      axes[j+1] = axes[j]
      j -= 1
    axes[j+1] = key

  if profile_enabled:
    t_tmp = time.perf_counter()
  if arr.dtype in (np.uint8, np.int8):
    _nd_pass_multi_odometer[uint8_t](<uint8_t*>datap, outp, <size_t>nd, cshape, cstrides, <size_t>axes[0], <float>anis[axes[0]], black_border, parallel)
  elif arr.dtype in (np.uint16, np.int16):
    _nd_pass_multi_odometer[uint16_t](<uint16_t*>datap, outp, <size_t>nd, cshape, cstrides, <size_t>axes[0], <float>anis[axes[0]], black_border, parallel)
  elif arr.dtype in (np.uint32, np.int32):
    _nd_pass_multi_odometer[uint32_t](<uint32_t*>datap, outp, <size_t>nd, cshape, cstrides, <size_t>axes[0], <float>anis[axes[0]], black_border, parallel)
  elif arr.dtype in (np.uint64, np.int64):
    _nd_pass_multi_odometer[uint64_t](<uint64_t*>datap, outp, <size_t>nd, cshape, cstrides, <size_t>axes[0], <float>anis[axes[0]], black_border, parallel)
  elif arr.dtype == np.float32:
    _nd_pass_multi_odometer[float](<float*>datap, outp, <size_t>nd, cshape, cstrides, <size_t>axes[0], <float>anis[axes[0]], black_border, parallel)
  elif arr.dtype == np.float64:
    _nd_pass_multi_odometer[double](<double*>datap, outp, <size_t>nd, cshape, cstrides, <size_t>axes[0], <float>anis[axes[0]], black_border, parallel)
  elif arr.dtype == bool:
    _nd_pass_multi_odometer[native_bool](<native_bool*>datap, outp, <size_t>nd, cshape, cstrides, <size_t>axes[0], <float>anis[axes[0]], black_border, parallel)
  if profile_enabled:
    multi_elapsed = time.perf_counter() - t_tmp
    profile_axes.append({'axis': int(axes[0]), 'kind': 'multi', 'time': multi_elapsed})

  if not black_border:
    if profile_enabled:
      t_tmp = time.perf_counter()
    mask_inf = np.isinf(out)
    if mask_inf.any():
      out[mask_inf] = np.finfo(np.float32).max
    if profile_enabled:
      profile_sections['multi_fix'] = time.perf_counter() - t_tmp
  elif profile_enabled:
    profile_sections['multi_fix'] = 0.0

  for a in range(1, nd):
    if profile_enabled:
      t_tmp = time.perf_counter()
    if arr.dtype in (np.uint8, np.int8):
      _nd_pass_parabolic_odometer[uint8_t](<uint8_t*>datap, outp, <size_t>nd, cshape, cstrides, <size_t>axes[a], <float>anis[axes[a]], black_border, parallel)
    elif arr.dtype in (np.uint16, np.int16):
      _nd_pass_parabolic_odometer[uint16_t](<uint16_t*>datap, outp, <size_t>nd, cshape, cstrides, <size_t>axes[a], <float>anis[axes[a]], black_border, parallel)
    elif arr.dtype in (np.uint32, np.int32):
      _nd_pass_parabolic_odometer[uint32_t](<uint32_t*>datap, outp, <size_t>nd, cshape, cstrides, <size_t>axes[a], <float>anis[axes[a]], black_border, parallel)
    elif arr.dtype in (np.uint64, np.int64):
      _nd_pass_parabolic_odometer[uint64_t](<uint64_t*>datap, outp, <size_t>nd, cshape, cstrides, <size_t>axes[a], <float>anis[axes[a]], black_border, parallel)
    elif arr.dtype == np.float32:
      _nd_pass_parabolic_odometer[float](<float*>datap, outp, <size_t>nd, cshape, cstrides, <size_t>axes[a], <float>anis[axes[a]], black_border, parallel)
    elif arr.dtype == np.float64:
      _nd_pass_parabolic_odometer[double](<double*>datap, outp, <size_t>nd, cshape, cstrides, <size_t>axes[a], <float>anis[axes[a]], black_border, parallel)
    elif arr.dtype == bool:
      _nd_pass_parabolic_odometer[native_bool](<native_bool*>datap, outp, <size_t>nd, cshape, cstrides, <size_t>axes[a], <float>anis[axes[a]], black_border, parallel)
    if profile_enabled:
      t_axis = time.perf_counter() - t_tmp
      parabolic_elapsed += t_axis
      profile_axes.append({'axis': int(axes[a]), 'kind': 'parabolic', 'time': t_axis})

  if not black_border:
    if profile_enabled:
      t_tmp = time.perf_counter()
    mask_max = out >= np.finfo(np.float32).max
    if mask_max.any():
      out[mask_max] = np.inf
    if profile_enabled:
      profile_sections['post_fix'] = time.perf_counter() - t_tmp
  elif profile_enabled:
    profile_sections['post_fix'] = 0.0

  free(cshape); free(cstrides); free(axes)
  if profile_enabled:
    profile_sections['multi_pass'] = multi_elapsed
    profile_sections['parabolic_pass'] = parabolic_elapsed
    profile_sections['total'] = time.perf_counter() - t_total_start
    _nd_profile_last = profile_data

  order_ch2 = 'C' if arr.flags.c_contiguous else 'F'
  return np.reshape(out, arr.shape, order=order_ch2)

@cython.binding(True)
def edtsq_nd_last_profile():
  """Return the last ND profile captured when EDT_ND_PROFILE is enabled."""
  return _nd_profile_last




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

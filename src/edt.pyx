# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
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
  feature_transform, expand_labels

Legacy dimension-specific APIs are available through ``edt.legacy`` when the
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
from libc.string cimport memcpy
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
import warnings


_ND_MIN_VOXELS_PER_THREAD_DEFAULT = 32768
_ND_MIN_LINES_PER_THREAD_DEFAULT = 32
_ND_2D_THRESHOLDS = (
  (128 * 128, 4),
  (512 * 512, 8),
)
_ND_2D_MAX_THREADS = 12
_ND_3D_THRESHOLDS = (
  (32, 4),
  (48, 8),
  (64, 12),
)
_ND_3D_MAX_THREADS = 16


_nd_profile_last = None

# --- voxel_graph gating helpers (C-order contiguous) ---
cdef inline void _gate_voxel_graph_uint8(
    const np.uint8_t* graph,
    const np.uint8_t* fg,
    np.uint8_t* dbl,
    size_t nd,
    const size_t* shape,
    const size_t* dbl_strides
) noexcept nogil:
  cdef size_t total = 1
  cdef size_t d, i, tmp, coord, base_even
  cdef size_t bit
  for d in range(nd):
    total *= shape[d]
  for i in range(total):
    tmp = i
    base_even = 0
    for d in range(nd-1, -1, -1):
      coord = tmp % shape[d]
      tmp //= shape[d]
      base_even += coord * (dbl_strides[d] * 2)
    for d in range(nd):
      bit = 2 * (nd - 1 - d)
      if (graph[i] >> bit) & 1:
        dbl[base_even + dbl_strides[d]] = fg[i]

cdef inline void _gate_voxel_graph_uint16(
    const np.uint16_t* graph,
    const np.uint8_t* fg,
    np.uint8_t* dbl,
    size_t nd,
    const size_t* shape,
    const size_t* dbl_strides
) noexcept nogil:
  cdef size_t total = 1
  cdef size_t d, i, tmp, coord, base_even
  cdef size_t bit
  for d in range(nd):
    total *= shape[d]
  for i in range(total):
    tmp = i
    base_even = 0
    for d in range(nd-1, -1, -1):
      coord = tmp % shape[d]
      tmp //= shape[d]
      base_even += coord * (dbl_strides[d] * 2)
    for d in range(nd):
      bit = 2 * (nd - 1 - d)
      if (graph[i] >> bit) & 1:
        dbl[base_even + dbl_strides[d]] = fg[i]

cdef inline void _gate_voxel_graph_uint32(
    const np.uint32_t* graph,
    const np.uint8_t* fg,
    np.uint8_t* dbl,
    size_t nd,
    const size_t* shape,
    const size_t* dbl_strides
) noexcept nogil:
  cdef size_t total = 1
  cdef size_t d, i, tmp, coord, base_even
  cdef size_t bit
  for d in range(nd):
    total *= shape[d]
  for i in range(total):
    tmp = i
    base_even = 0
    for d in range(nd-1, -1, -1):
      coord = tmp % shape[d]
      tmp //= shape[d]
      base_even += coord * (dbl_strides[d] * 2)
    for d in range(nd):
      bit = 2 * (nd - 1 - d)
      if (graph[i] >> bit) & 1:
        dbl[base_even + dbl_strides[d]] = fg[i]

cdef inline void _gate_voxel_graph_uint64(
    const np.uint64_t* graph,
    const np.uint8_t* fg,
    np.uint8_t* dbl,
    size_t nd,
    const size_t* shape,
    const size_t* dbl_strides
) noexcept nogil:
  cdef size_t total = 1
  cdef size_t d, i, tmp, coord, base_even
  cdef size_t bit
  for d in range(nd):
    total *= shape[d]
  for i in range(total):
    tmp = i
    base_even = 0
    for d in range(nd-1, -1, -1):
      coord = tmp % shape[d]
      tmp //= shape[d]
      base_even += coord * (dbl_strides[d] * 2)
    for d in range(nd):
      bit = 2 * (nd - 1 - d)
      if (graph[i] >> bit) & 1:
        dbl[base_even + dbl_strides[d]] = fg[i]

cdef inline void _gate_voxel_graph_labels_u8(
    const np.uint8_t* graph,
    const np.uint32_t* labels,
    np.uint32_t* dbl,
    size_t nd,
    const size_t* shape,
    const size_t* dbl_strides
) noexcept nogil:
  cdef size_t total = 1
  cdef size_t d, i, tmp, coord, base_even
  cdef size_t bit
  for d in range(nd):
    total *= shape[d]
  for i in range(total):
    tmp = i
    base_even = 0
    for d in range(nd-1, -1, -1):
      coord = tmp % shape[d]
      tmp //= shape[d]
      base_even += coord * (dbl_strides[d] * 2)
    for d in range(nd):
      bit = 2 * (nd - 1 - d)
      if (graph[i] >> bit) & 1:
        dbl[base_even + dbl_strides[d]] = labels[i]

cdef inline void _gate_voxel_graph_labels_u16(
    const np.uint16_t* graph,
    const np.uint32_t* labels,
    np.uint32_t* dbl,
    size_t nd,
    const size_t* shape,
    const size_t* dbl_strides
) noexcept nogil:
  cdef size_t total = 1
  cdef size_t d, i, tmp, coord, base_even
  cdef size_t bit
  for d in range(nd):
    total *= shape[d]
  for i in range(total):
    tmp = i
    base_even = 0
    for d in range(nd-1, -1, -1):
      coord = tmp % shape[d]
      tmp //= shape[d]
      base_even += coord * (dbl_strides[d] * 2)
    for d in range(nd):
      bit = 2 * (nd - 1 - d)
      if (graph[i] >> bit) & 1:
        dbl[base_even + dbl_strides[d]] = labels[i]

cdef inline void _gate_voxel_graph_labels_u32(
    const np.uint32_t* graph,
    const np.uint32_t* labels,
    np.uint32_t* dbl,
    size_t nd,
    const size_t* shape,
    const size_t* dbl_strides
) noexcept nogil:
  cdef size_t total = 1
  cdef size_t d, i, tmp, coord, base_even
  cdef size_t bit
  for d in range(nd):
    total *= shape[d]
  for i in range(total):
    tmp = i
    base_even = 0
    for d in range(nd-1, -1, -1):
      coord = tmp % shape[d]
      tmp //= shape[d]
      base_even += coord * (dbl_strides[d] * 2)
    for d in range(nd):
      bit = 2 * (nd - 1 - d)
      if (graph[i] >> bit) & 1:
        dbl[base_even + dbl_strides[d]] = labels[i]

cdef inline void _gate_voxel_graph_labels_u64(
    const np.uint64_t* graph,
    const np.uint32_t* labels,
    np.uint32_t* dbl,
    size_t nd,
    const size_t* shape,
    const size_t* dbl_strides
) noexcept nogil:
  cdef size_t total = 1
  cdef size_t d, i, tmp, coord, base_even
  cdef size_t bit
  for d in range(nd):
    total *= shape[d]
  for i in range(total):
    tmp = i
    base_even = 0
    for d in range(nd-1, -1, -1):
      coord = tmp % shape[d]
      tmp //= shape[d]
      base_even += coord * (dbl_strides[d] * 2)
    for d in range(nd):
      bit = 2 * (nd - 1 - d)
      if (graph[i] >> bit) & 1:
        dbl[base_even + dbl_strides[d]] = labels[i]


# --- Barrier graph building/conversion helpers ---

def _build_barrier_graph_from_labels(labels, int parallel=1):
  """
  Build a barrier graph from labels for standard EDT (no voxel_graph).

  All foreground voxels get the foreground marker (0x80 for <=4D, 0x8000 for 5D+).
  Adjacent foreground voxels with the same label get forward edge bits.

  This allows using the barrier EDT algorithm for standard (non-voxel_graph) EDT.
  Uses fast C++ implementation with optional parallelization.
  """
  cdef int ndim = labels.ndim
  cdef int bits_needed = 2 * ndim

  if bits_needed <= 8:
    graph_dtype = np.uint8
  else:
    graph_dtype = np.uint16

  # Ensure labels is contiguous and get supported dtype
  if not labels.flags.c_contiguous:
    labels = np.ascontiguousarray(labels)

  cdef np.ndarray[int64_t, ndim=1] shape_arr = np.array(labels.shape, dtype=np.int64)
  cdef int64_t* shape_ptr = <int64_t*> np.PyArray_DATA(shape_arr)

  graph = np.empty(labels.shape, dtype=graph_dtype)

  cdef void* labels_ptr = np.PyArray_DATA(labels)
  cdef void* graph_ptr = np.PyArray_DATA(graph)
  cdef int ndim_c = ndim
  cdef int parallel_c = parallel

  # Call C++ implementation based on label dtype
  if labels.dtype == np.uint8:
    if graph_dtype == np.uint8:
      with nogil:
        build_barrier_graph_from_labels_cpp[uint8_t, uint8_t](
          <const uint8_t*>labels_ptr, <uint8_t*>graph_ptr, shape_ptr, ndim_c, parallel_c)
    else:
      with nogil:
        build_barrier_graph_from_labels_cpp[uint8_t, uint16_t](
          <const uint8_t*>labels_ptr, <uint16_t*>graph_ptr, shape_ptr, ndim_c, parallel_c)
  elif labels.dtype == np.uint16:
    if graph_dtype == np.uint8:
      with nogil:
        build_barrier_graph_from_labels_cpp[uint16_t, uint8_t](
          <const uint16_t*>labels_ptr, <uint8_t*>graph_ptr, shape_ptr, ndim_c, parallel_c)
    else:
      with nogil:
        build_barrier_graph_from_labels_cpp[uint16_t, uint16_t](
          <const uint16_t*>labels_ptr, <uint16_t*>graph_ptr, shape_ptr, ndim_c, parallel_c)
  elif labels.dtype == np.uint32:
    if graph_dtype == np.uint8:
      with nogil:
        build_barrier_graph_from_labels_cpp[uint32_t, uint8_t](
          <const uint32_t*>labels_ptr, <uint8_t*>graph_ptr, shape_ptr, ndim_c, parallel_c)
    else:
      with nogil:
        build_barrier_graph_from_labels_cpp[uint32_t, uint16_t](
          <const uint32_t*>labels_ptr, <uint16_t*>graph_ptr, shape_ptr, ndim_c, parallel_c)
  elif labels.dtype == np.uint64:
    if graph_dtype == np.uint8:
      with nogil:
        build_barrier_graph_from_labels_cpp[uint64_t, uint8_t](
          <const uint64_t*>labels_ptr, <uint8_t*>graph_ptr, shape_ptr, ndim_c, parallel_c)
    else:
      with nogil:
        build_barrier_graph_from_labels_cpp[uint64_t, uint16_t](
          <const uint64_t*>labels_ptr, <uint16_t*>graph_ptr, shape_ptr, ndim_c, parallel_c)
  else:
    # Fallback for other dtypes - convert to uint32
    labels_u32 = labels.astype(np.uint32, copy=False)
    labels_ptr = np.PyArray_DATA(labels_u32)
    if graph_dtype == np.uint8:
      with nogil:
        build_barrier_graph_from_labels_cpp[uint32_t, uint8_t](
          <const uint32_t*>labels_ptr, <uint8_t*>graph_ptr, shape_ptr, ndim_c, parallel_c)
    else:
      with nogil:
        build_barrier_graph_from_labels_cpp[uint32_t, uint16_t](
          <const uint32_t*>labels_ptr, <uint16_t*>graph_ptr, shape_ptr, ndim_c, parallel_c)

  return graph


def _edtsq_barrier_direct(graph, anisotropy=None, int parallel=1):
  """
  Direct barrier EDT call with pre-built graph (minimal wrapper).

  This bypasses graph building and most Python overhead.
  Used for benchmarking the raw EDT performance.
  """
  cdef int ndim_c = graph.ndim
  if anisotropy is None:
    anisotropy = (1.0,) * ndim_c
  if len(anisotropy) != ndim_c:
    raise ValueError("anisotropy length must match ndim")

  if not graph.flags.c_contiguous:
    graph = np.ascontiguousarray(graph)

  cdef np.ndarray[int64_t, ndim=1] shape_arr = np.array(graph.shape, dtype=np.int64)
  cdef np.ndarray[np.float32_t, ndim=1] aniso_arr = np.array(anisotropy, dtype=np.float32)
  cdef int64_t voxels = graph.size
  cdef np.ndarray[np.float32_t, ndim=1] out_flat = np.empty(voxels, dtype=np.float32)

  cdef float* result_ptr = NULL
  cdef int64_t* shape_ptr = <int64_t*> np.PyArray_DATA(shape_arr)
  cdef float* aniso_ptr = <float*> np.PyArray_DATA(aniso_arr)
  cdef void* graph_ptr = np.PyArray_DATA(graph)

  cdef bint is_u8 = graph.dtype == np.uint8
  cdef bint is_u16 = graph.dtype == np.uint16

  if is_u8:
    with nogil:
      result_ptr = edtsq_barrier[uint8_t](
        <const uint8_t*>graph_ptr, shape_ptr, aniso_ptr, ndim_c, parallel)
  elif is_u16:
    with nogil:
      result_ptr = edtsq_barrier[uint16_t](
        <const uint16_t*>graph_ptr, shape_ptr, aniso_ptr, ndim_c, parallel)
  else:
    raise TypeError("graph must be uint8 or uint16")

  cdef int64_t i
  for i in range(voxels):
    out_flat[i] = result_ptr[i]

  delete_array_barrier[float](result_ptr)
  return out_flat.reshape(graph.shape)


def _voxel_graph_to_barrier_graph(labels, voxel_graph):
  """
  Convert bidirectional voxel_graph to barrier_graph format.

  The standard voxel_graph format uses 2*ndim bits per voxel:
  - For each axis a: positive direction at bit (2*(ndim-1-a)+0)
  - For each axis a: negative direction at bit (2*(ndim-1-a)+1)

  The barrier_graph format uses forward edges only + foreground marker:
  - Forward edge for axis a at bit (2*(ndim-1-a))
  - High bit (0x80 for uint8, 0x8000 for uint16) marks foreground
  """
  if voxel_graph.shape != labels.shape:
    raise ValueError("voxel_graph shape must match labels")

  cdef int ndim = voxel_graph.ndim
  cdef int bits_needed = 2 * ndim

  if bits_needed <= 8:
    dtype = np.uint8
    fg_marker = 0x80
  else:
    dtype = np.uint16
    fg_marker = 0x8000

  # Build mask for positive direction bits only
  # Positive direction for axis a is at bit (2*(ndim-1-a)+0)
  # which equals the barrier forward edge position
  cdef int pos_mask = 0
  cdef int ax
  for ax in range(ndim):
    pos_mask |= 1 << (2 * (ndim - 1 - ax))

  # Extract only positive direction bits
  barrier_graph = (voxel_graph.astype(np.uint32) & pos_mask).astype(dtype)

  # Add foreground marker to foreground voxels
  barrier_graph[labels != 0] |= fg_marker

  return barrier_graph


def _env_int(name: str, default: int) -> int:
  value = os.environ.get(name)
  if value is None:
    return default
  try:
    return int(value)
  except Exception:
    return default


cdef inline int _apply_thresholds(int measure, tuple thresholds, int default_cap, int parallel):
  cdef int capped = parallel
  cdef int limit
  cdef int cap
  for threshold in thresholds:
    limit = <int>threshold[0]
    cap = <int>threshold[1]
    if measure <= limit:
      if cap > 0 and capped > cap:
        capped = cap
      return capped
  if default_cap > 0 and capped > default_cap:
    capped = default_cap
  return capped



class _LegacyModuleProxy(types.ModuleType):
  """Lazy loader for the legacy dimension-specific EDT extension."""

  def __init__(self):
    super().__init__('edt.legacy')
    self._module = None
    self._load_error = None

  def _candidate_roots(self):
    for env_key in ('EDT_LEGACY_PATH', 'EDT_ORIGINAL_PATH'):
      env_path = os.environ.get(env_key)
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
    patterns = ('**/edt_legacy*.so', '**/edt_legacy*.pyd', '**/edt_legacy*.dll',
                '**/edt_original*.so', '**/edt_original*.pyd', '**/edt_original*.dll')
    for root in roots:
      for pattern in patterns:
        for path in sorted(root.glob(pattern), reverse=True):
          if path.is_file():
            yield path

  def _register(self, module):
    module.__name__ = 'edt.legacy'
    module.available = lambda: True
    sys.modules['edt.legacy'] = module
    # Maintain backwards compatibility for any lingering imports.
    sys.modules.setdefault('edt.original', module)
    self._module = module
    self._load_error = None
    return module

  def _load(self):
    if self._module is not None:
      return self._module
    if self._load_error is not None:
      raise self._load_error

    errors = []
    for module_name in ('edt_legacy', 'edt_original'):
      try:
        module = importlib.import_module(module_name)
        return self._register(module)
      except Exception as exc:
        errors.append((f'import:{module_name}', exc))

    # fall back to disk lookup (for developers with external clone)
    roots = list(self._candidate_roots())
    candidates = list(self._iter_extension_candidates(roots))

    for candidate in candidates:
      try:
        spec = importlib.util.spec_from_file_location('edt_legacy_native', candidate)
        if spec is None or spec.loader is None:
          raise ImportError(f'Unable to create loader for {candidate}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return self._register(module)
      except Exception as exc:
        errors.append((str(candidate), exc))

    if errors:
      messages = ', '.join(f"{source}: {exc}" for source, exc in errors)
      err = ImportError(f'Failed to load legacy EDT extension ({messages})')
      self._load_error = err
      raise err

    if roots:
      hint = roots[0]
      err = ImportError(
        f'Legacy EDT extension not built in {hint}. '
        'Run `pip install -e .` there or build the packaged edt.legacy._native.'
      )
      self._load_error = err
      raise err
    err = ImportError(
      "Legacy EDT implementation not available. Install the package with the"
      " bundled edt.legacy module or provide a build in ./original_repo."
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


legacy = _LegacyModuleProxy()


class _NdV1ModuleProxy(types.ModuleType):
  """Lazy loader for the ND V1 EDT extension (pre-barrier implementation)."""

  def __init__(self):
    super().__init__('edt.nd_v1')
    self._module = None
    self._load_error = None

  def _candidate_roots(self):
    env_path = os.environ.get('EDT_ND_V1_PATH')
    if env_path:
      path = pathlib.Path(env_path).expanduser()
      if path.exists():
        yield path
    base = pathlib.Path(__file__).resolve()
    seen = set()
    for parent in (base.parent,) + tuple(base.parents):
      candidate = (parent / 'nd_v1').resolve()
      if candidate.exists() and candidate not in seen:
        seen.add(candidate)
        yield candidate

  def _iter_extension_candidates(self, roots):
    patterns = ('**/edt_nd_v1*.so', '**/edt_nd_v1*.pyd', '**/edt_nd_v1*.dll')
    for root in roots:
      for pattern in patterns:
        for path in sorted(root.glob(pattern), reverse=True):
          if path.is_file():
            yield path

  def _register(self, module):
    module.__name__ = 'edt.nd_v1'
    module.available = lambda: True
    sys.modules['edt.nd_v1'] = module
    self._module = module
    self._load_error = None
    return module

  def _load(self):
    if self._module is not None:
      return self._module
    if self._load_error is not None:
      raise self._load_error

    errors = []
    for module_name in ('edt_nd_v1',):
      try:
        module = importlib.import_module(module_name)
        return self._register(module)
      except Exception as exc:
        errors.append((f'import:{module_name}', exc))

    # fall back to disk lookup
    roots = list(self._candidate_roots())
    candidates = list(self._iter_extension_candidates(roots))

    for candidate in candidates:
      try:
        spec = importlib.util.spec_from_file_location('edt_nd_v1_native', candidate)
        if spec is None or spec.loader is None:
          raise ImportError(f'Unable to create loader for {candidate}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return self._register(module)
      except Exception as exc:
        errors.append((str(candidate), exc))

    if errors:
      messages = ', '.join(f"{source}: {exc}" for source, exc in errors)
      err = ImportError(f'Failed to load ND V1 EDT extension ({messages})')
      self._load_error = err
      raise err

    err = ImportError(
      "ND V1 EDT implementation not available. Build it from the nd_v1/ directory."
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
    """Return True if the ND V1 module could be loaded."""
    if self._module is not None:
      return True
    try:
      self._load()
    except Exception:
      return False
    return True


nd_v1 = _NdV1ModuleProxy()


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

@cython.binding(True)
def nd_tuning_force(tile, prefetch_step, chunks_per_thread):
  """Force ND tuning settings, including zero values."""
  cdef size_t t = <size_t>int(tile)
  cdef size_t p = <size_t>int(prefetch_step)
  cdef size_t c = <size_t>int(chunks_per_thread)
  with nogil:
    nd_set_tuning_force(t, p, c)

@cython.binding(True)
def nd_multi_batch(batch):
  """Set ND multi-seg batch size (1-32)."""
  cdef size_t b = <size_t>int(batch)
  with nogil:
    nd_set_multi_batch(b)

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
from libc.stdint cimport int64_t

# Barrier EDT (primary voxel_graph path)
cdef extern from "nd_barrier_core.hpp" namespace "nd_barrier":
  cdef float* edtsq_barrier[GRAPH_T](
    const GRAPH_T* graph,
    const int64_t* shape,
    const float* anisotropy,
    int ndim,
    int parallel
  ) nogil

  cdef void build_barrier_graph_from_labels_cpp "nd_barrier::build_barrier_graph_from_labels"[LABEL_T, GRAPH_T](
    const LABEL_T* labels,
    GRAPH_T* graph,
    const int64_t* shape,
    int ndim,
    int parallel
  ) nogil

cdef extern from *:
  """
  template<typename T>
  void delete_array_barrier(T* ptr) { delete[] ptr; }
  """
  void delete_array_barrier[T](T* ptr)

cdef extern from "edt.hpp" namespace "pyedt":
    cdef void nd_set_base_block(size_t block) nogil
    cdef void nd_set_multi_batch(size_t batch) nogil
    cdef void nd_set_tuning_force(size_t tile, size_t prefetch_step, size_t chunks_per_thread) nogil
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
    cdef bint _nd_pass_multi_compiled[T](
        T* labels, float* dest,
        size_t dims, size_t* shape, size_t* strides,
        size_t ax, float anis, native_bool black_border, int parallel
    ) nogil
    cdef bint _nd_pass_parabolic_compiled[T](
        T* labels, float* dest,
        size_t dims, size_t* shape, size_t* strides,
        size_t ax, float anis, native_bool black_border, int parallel
    ) nogil
    cdef void squared_edt_1d_parabolic_multi_seg[T](
        T* segids, float* f,
        int n, long stride, float anisotropy,
        native_bool black_border
    ) nogil
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
    cdef void tofinite(float* data, size_t voxels) nogil
    cdef void toinfinite(float* data, size_t voxels) nogil

ctypedef fused nd_t:
  uint8_t
  uint16_t
  uint32_t
  uint64_t
  float
  double
  native_bool

cdef void _run_multi_pass(nd_t* datap, float* outp,
                          size_t nd, size_t* cshape, size_t* cstrides,
                          size_t axis, float anisotropy,
                          native_bool black_border, int parallel) noexcept nogil:
  cdef bint used = _nd_pass_multi_compiled(
      datap, outp, nd, cshape, cstrides, axis, anisotropy, black_border, parallel
  )
  if not used:
    _nd_pass_multi_odometer(
        datap, outp, nd, cshape, cstrides, axis, anisotropy, black_border, parallel
    )

cdef void _run_parabolic_pass(nd_t* datap, float* outp,
                              size_t nd, size_t* cshape, size_t* cstrides,
                              size_t axis, float anisotropy,
                              native_bool black_border, int parallel) noexcept nogil:
  cdef bint used = _nd_pass_parabolic_compiled(
      datap, outp, nd, cshape, cstrides, axis, anisotropy, black_border, parallel
  )
  if not used:
    _nd_pass_parabolic_odometer(
        datap, outp, nd, cshape, cstrides, axis, anisotropy, black_border, parallel
    )

cpdef np.ndarray _edtsq_nd_typed(
    nd_t[::1] data,
    tuple shape,
    tuple anis,
    bint c_contig,
    native_bool black_border,
    int parallel,
    bint profile_enabled,
    dict profile_sections,
    list profile_axes,
    object debug_stage,
  ):
  cdef Py_ssize_t nd = len(shape)
  cdef size_t* cshape = <size_t*> malloc(nd * sizeof(size_t))
  cdef size_t* cstrides = <size_t*> malloc(nd * sizeof(size_t))
  cdef float* canis = <float*> malloc(nd * sizeof(float))
  if cshape == NULL or cstrides == NULL or canis == NULL:
    if cshape != NULL:
      free(cshape)
    if cstrides != NULL:
      free(cstrides)
    if canis != NULL:
      free(canis)
    raise MemoryError('Allocation failure')

  cdef size_t total = 1
  cdef Py_ssize_t i
  for i in range(nd):
    cshape[i] = <size_t>shape[i]
    canis[i] = <float>anis[i]
  if c_contig:
    cstrides[nd-1] = 1
    for i in range(nd-2, -1, -1):
      cstrides[i] = cstrides[i+1] * cshape[i+1]
  else:
    cstrides[0] = 1
    for i in range(1, nd):
      cstrides[i] = cstrides[i-1] * cshape[i-1]
  for i in range(nd):
    total *= cshape[i]

  cdef np.ndarray out = np.zeros((<Py_ssize_t> total,), dtype=np.float32)
  cdef float* outp = <float*> np.PyArray_DATA(out)
  cdef nd_t* datap = &data[0]

  cdef Py_ssize_t* axes = <Py_ssize_t*> malloc(nd * sizeof(Py_ssize_t))
  if axes == NULL:
    free(cshape); free(cstrides); free(canis)
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

  cdef double t_tmp
  cdef double t_axis
  cdef double multi_elapsed = 0.0
  cdef double parabolic_elapsed = 0.0

  if profile_enabled:
    t_tmp = time.perf_counter()
  with nogil:
    _run_multi_pass(datap, outp, <size_t>nd, cshape, cstrides,
                    <size_t>axes[0], canis[axes[0]], black_border, parallel)
  if profile_enabled:
    multi_elapsed = time.perf_counter() - t_tmp
    profile_axes.append({'axis': int(axes[0]), 'kind': 'multi', 'time': multi_elapsed})

  if debug_stage == 'multi':
    free(cshape); free(cstrides); free(canis); free(axes)
    order_ch2 = 'C' if c_contig else 'F'
    return np.reshape(out, shape, order=order_ch2)

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
    with nogil:
      _run_parabolic_pass(datap, outp, <size_t>nd, cshape, cstrides,
                          <size_t>axes[a], canis[axes[a]], black_border, parallel)
    if profile_enabled:
      t_axis = time.perf_counter() - t_tmp
      parabolic_elapsed += t_axis
      profile_axes.append({'axis': int(axes[a]), 'kind': 'parabolic', 'time': t_axis})

  if debug_stage == 'parabolic':
    free(cshape); free(cstrides); free(canis); free(axes)
    order_ch2 = 'C' if c_contig else 'F'
    return np.reshape(out, shape, order=order_ch2)

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

  free(cshape); free(cstrides); free(canis); free(axes)
  if profile_enabled:
    profile_sections['multi_pass'] = multi_elapsed
    profile_sections['parabolic_pass'] = parabolic_elapsed

  order_ch2 = 'C' if c_contig else 'F'
  return np.reshape(out, shape, order=order_ch2)


cpdef void _parabolic_1d_typed(
    nd_t[::1] data,
    float[::1] vals,
    int n,
    long stride,
    float anisotropy,
    native_bool black_border,
  ):
  cdef nd_t* datap = &data[0]
  cdef float* valsp = &vals[0]
  with nogil:
    squared_edt_1d_parabolic_multi_seg(datap, valsp, n, stride, anisotropy, black_border)

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


def _adaptive_thread_limit_nd(parallel, shape, requested=None):
  """General ND limiter that matches 2D/3D heuristics and scales for higher dims."""
  parallel = max(1, parallel)
  try:
    adapt_enabled = bool(int(os.environ.get('EDT_ADAPTIVE_THREADS', '1')))
  except Exception:
    adapt_enabled = True
  if not adapt_enabled:
    return parallel

  dims = len(shape)
  if dims <= 1:
    return parallel

  capped = parallel

  if dims == 2:
    total = <int>(shape[0] * shape[1])
    capped = _apply_thresholds(total, _ND_2D_THRESHOLDS, _ND_2D_MAX_THREADS, capped)
    return max(1, capped)

  if dims == 3:
    longest = <int>max(shape)
    capped = _apply_thresholds(longest, _ND_3D_THRESHOLDS, _ND_3D_MAX_THREADS, capped)
    return max(1, capped)

  min_voxels = _env_int('EDT_ND_MIN_VOXELS_PER_THREAD', _ND_MIN_VOXELS_PER_THREAD_DEFAULT)
  if min_voxels < 1:
    min_voxels = _ND_MIN_VOXELS_PER_THREAD_DEFAULT
  min_lines = _env_int('EDT_ND_MIN_LINES_PER_THREAD', _ND_MIN_LINES_PER_THREAD_DEFAULT)
  if min_lines < 1:
    min_lines = _ND_MIN_LINES_PER_THREAD_DEFAULT

  total = 1
  cdef int extent
  for extent in shape:
    total *= extent
  longest = max(shape)
  lines = max(1, total // longest)

  cap_lines = max(1, lines // min_lines)
  cap_voxels = max(1, total // min_voxels)
  cap = max(cap_lines, cap_voxels)
  capped = min(capped, cap)
  return max(1, capped)

@cython.binding(True)
def expand_labels(
    data, anisotropy=None,
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
  parallel : int, optional
      Number of threads; if <= 0, uses cpu_count().
  voxel_graph : ndarray, optional
      Not supported for expand_labels; only distance transforms accept
      voxel_graph. Passing a non-None value raises ValueError.
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
  if voxel_graph is not None:
    raise ValueError("voxel_graph is only supported for distance transforms (edtsq/edt)")
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
  cdef size_t* vg_shape
  cdef size_t* dbl_strides
  cdef void* graph_data
  cdef void* labels_data
  cdef void* dbl_data
  cdef Py_ssize_t a
  cdef Py_ssize_t nd

  arr = np.require(data, dtype=np.uint32, requirements='C')
  dims = arr.ndim
  nd = dims
  cdef tuple anis
  if anisotropy is None:
    anis = (1.0,) * dims
  else:
    anis = tuple(anisotropy) if hasattr(anisotropy, '__len__') else (float(anisotropy),) * dims
    if len(anis) != dims:
      raise ValueError('anisotropy length must match data.ndim')

  cdef int cpu_cap = 1
  try:
    cpu_cap = multiprocessing.cpu_count()
  except Exception:
    cpu_cap = max(1, parallel)

  if parallel <= 0:
    parallel = cpu_cap
  else:
    parallel = max(1, min(parallel, cpu_cap))

  if dims == 1:
    # reuse 1D midpoint selection from expand_labels 1D branch
    n1 = arr.shape[0]
    out1 = np.empty((n1,), dtype=np.uint32)
    seed_pos = np.flatnonzero(arr)
    if seed_pos.size == 0:
      out1.fill(0)
      if return_features:
        return out1, np.zeros((n1,), dtype=np.uintp)
      return out1
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
  cdef native_bool bb = False
  # Work with C-contiguous buffer
  if not arr.flags.c_contiguous:
    arr = np.ascontiguousarray(arr)
  cdef np.ndarray[np.uint8_t, ndim=1] seeds_flat = (arr.ravel(order='K') != 0).astype(np.uint8, order='C')
  cdef np.ndarray[np.uint32_t, ndim=1] labels_flat = arr.ravel(order='K').astype(np.uint32, order='C')
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
    raise MemoryError('Allocation failure in expand_labels')
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
  return lab_prev.reshape(arr.shape)


@cython.binding(True)
def feature_transform(
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
  voxel_graph : ndarray, optional
      Not supported for feature_transform; only distance transforms accept
      voxel_graph. Passing a non-None value raises ValueError.
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
  if voxel_graph is not None:
    raise ValueError("voxel_graph is only supported for distance transforms (edtsq/edt)")
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
  labels, feats = expand_labels(arr.astype(np.uint32, copy=False), anis, parallel, None, True)

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
    dist = edtsq_nd((arr != 0).astype(np.uint8, copy=False), anis, black_border, parallel, None)
    return feats, dist
  return feats


@cython.binding(True)
def edtsq(
    data, anisotropy=None, native_bool black_border=False,
    int parallel=1, voxel_graph=None, order=None,
):
  """
  Computes the squared anisotropic Euclidean Distance Transform (EDT) of
  N-dimensional numpy arrays.

  Squaring avoids the sqrt operation, so this can be faster if squared
  distances are acceptable.

  The input is made C-contiguous if needed. If you pass a Fortran-ordered
  array, it will be copied to C order; anisotropy is always specified in
  axis order (length == data.ndim).

  Supported dtypes:
    (u)int8, (u)int16, (u)int32, (u)int64,
    float32, float64, and boolean

  Parameters
  ----------
  data : ndarray
      N-dimensional input array (any supported dtype).
  anisotropy : float or sequence of float, optional
      Per-axis voxel size (default 1.0 for all axes).
  black_border : bool, optional
      If True, treat the border as background (default False).
  parallel : int, optional
      Number of threads; if <= 0, uses cpu_count().
  voxel_graph : ndarray, optional
      Per-voxel bitfield describing allowed +axis connections. Bit layout
      matches legacy (x/y/z ordering): bit = 1 << (2*(ndim-1-axis)).
      For 2D: axis0 uses bit 4, axis1 uses bit 1. For 3D: axis0 uses bit 16,
      axis1 uses bit 4, axis2 uses bit 1. If provided, distances respect the
      voxel-graph barriers using the legacy doubling scheme generalized to ND.
  order : ignored
      Retained for backwards compatibility.

  Returns
  -------
  ndarray, float32
      Squared Euclidean distances for each voxel.
  """
  return edtsq_nd(data, anisotropy, black_border, parallel, voxel_graph, order)


@cython.binding(True)
def edt(
    data, anisotropy=None, native_bool black_border=False,
    int parallel=1, voxel_graph=None, order=None,
  ):
  """
  Computes the anisotropic Euclidean Distance Transform (EDT) of
  N-dimensional numpy arrays.

  The input is made C-contiguous if needed. If you pass a Fortran-ordered
  array, it will be copied to C order; anisotropy is always specified in
  axis order (length == data.ndim).

  Supported dtypes:
    (u)int8, (u)int16, (u)int32, (u)int64,
    float32, float64, and boolean

  Parameters
  ----------
  data : ndarray
      N-dimensional input array (any supported dtype).
  anisotropy : float or sequence of float, optional
      Per-axis voxel size (default 1.0 for all axes).
  black_border : bool, optional
      If True, treat the border as background (default False).
  parallel : int, optional
      Number of threads; if <= 0, uses cpu_count().
  voxel_graph : ndarray, optional
      Per-voxel bitfield describing allowed +axis connections. Bit layout
      matches legacy (x/y/z ordering): bit = 1 << (2*(ndim-1-axis)).
      For 2D: axis0 uses bit 4, axis1 uses bit 1. For 3D: axis0 uses bit 16,
      axis1 uses bit 4, axis2 uses bit 1. If provided, distances respect the
      voxel-graph barriers using the legacy doubling scheme generalized to ND.
  order : ignored
      Retained for backwards compatibility.

  Returns
  -------
  ndarray, float32
      Euclidean distances for each voxel.
  """
  dt = edtsq_nd(data, anisotropy, black_border, parallel, voxel_graph, order)
  return np.sqrt(dt, dt)

def _set_nd_aliases():
  setattr(sys.modules[__name__], 'edtsq', edtsq)
  setattr(sys.modules[__name__], 'edt', edt)

_set_nd_aliases()
def __getattr__(name):
  """Forward unknown attributes to the packaged legacy module."""
  if name == 'edt':
    obj = globals().get('edt')
    if obj is not None:
      setattr(sys.modules[__name__], 'edt', obj)
      return obj
  if name == 'edtsq':
    obj = globals().get('edtsq')
    if obj is not None:
      setattr(sys.modules[__name__], 'edtsq', obj)
      return obj
  if name == 'legacy':
    setattr(sys.modules[__name__], 'legacy', legacy)
    return legacy
  if name == 'original':
    warnings.warn("`edt.original` is deprecated; use `edt.legacy` instead.", DeprecationWarning, stacklevel=2)
    setattr(sys.modules[__name__], 'original', legacy)
    return legacy
  if name == 'nd_v1':
    setattr(sys.modules[__name__], 'nd_v1', nd_v1)
    return nd_v1
  try:
    attr = getattr(legacy, name)
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
  """
  General ND squared EDT using barrier-aware algorithm.

  This is the core ND implementation used by `edtsq` and `edt`.
  The input is made C-contiguous if needed; anisotropy is always specified
  in axis order (length == data.ndim).

  Uses the barrier EDT algorithm which processes edge connectivity directly
  without grid doubling. For standard EDT (no voxel_graph), a barrier graph
  is built from labels where same-label adjacent voxels are connected.
  For voxel_graph inputs, the bidirectional graph is converted to barrier
  format (forward edges only + foreground marker).

  The previous ND implementation is available via edt.nd_v1 if built.
  """
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
  cdef Py_ssize_t nd
  cdef size_t* vg_shape
  cdef size_t* dbl_strides
  cdef void* graph_data
  cdef void* fg_data
  cdef void* dbl_data

  if isinstance(data, list):
    arr = np.array(data)
  elif isinstance(data, np.ndarray):
    arr = data
  else:
    arr = np.asarray(data)

  profile_env = os.environ.get('EDT_ND_PROFILE')
  if profile_env:
    try:
      profile_enabled = bool(int(profile_env))
    except Exception:
      profile_enabled = True
  if profile_enabled:
    t_total_start = time.perf_counter()

  cdef int base_block = _env_int('EDT_ND_BASE_BLOCK', 0)
  if base_block > 0:
    nd_set_base_block(<size_t>base_block)

  requested_parallel = parallel

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
  if not arr.flags.c_contiguous:
    arr = np.ascontiguousarray(arr)

  dims = arr.ndim
  nd = dims
  cdef tuple anis
  if anisotropy is None:
    anis = (1.0,) * dims
  else:
    anis = tuple(anisotropy) if hasattr(anisotropy, '__len__') else (float(anisotropy),) * dims
    if len(anis) != dims:
      raise ValueError('anisotropy length must match data.ndim')
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

  parallel = _adaptive_thread_limit_nd(p, arr.shape, requested_parallel)

  if profile_enabled:
    profile_data['parallel_requested'] = requested_parallel
    profile_data['parallel_used'] = parallel

  # --- Barrier EDT path (primary implementation) ---
  # Both voxel_graph and standard EDT now use barrier algorithm
  cdef np.ndarray barrier_graph
  cdef np.ndarray[int64_t, ndim=1] shape_arr
  cdef np.ndarray[np.float32_t, ndim=1] anis_arr
  cdef int64_t* shape_ptr
  cdef float* anis_ptr
  cdef void* barrier_ptr
  cdef float* result_ptr
  cdef int64_t voxels
  cdef int ndim_c = <int>dims
  cdef int parallel_c = <int>parallel
  cdef int64_t ii

  if voxel_graph is not None:
    # Use grid doubling approach for voxel_graph to match legacy behavior
    graph = np.asarray(voxel_graph)
    if graph.shape != arr.shape:
      raise ValueError('voxel_graph must have the same shape as data')
    if not graph.flags.c_contiguous:
      graph = np.ascontiguousarray(graph)
    graph_u = graph.astype(np.uint64, copy=False)
    req_bits = 2 * dims
    if req_bits > 64:
      raise ValueError('voxel_graph dtype does not have enough bits for data.ndim')

    fg = (arr != 0).astype(np.uint8, copy=False)

    # Build doubled grid and populate all even + multi-odd positions.
    dbl_shape = tuple(int(s * 2) for s in fg.shape)
    dbl = np.zeros(dbl_shape, dtype=fg.dtype)
    for combo in range(1 << dims):
      # Skip single-axis odd positions; they are gated by voxel_graph below.
      if combo != 0 and (combo & (combo - 1)) == 0:
        continue
      slc = []
      for ax in range(dims):
        slc.append(slice(1, None, 2) if (combo & (1 << ax)) else slice(0, None, 2))
      dbl[tuple(slc)] = fg

    vg_shape = <size_t*> malloc(dims * sizeof(size_t))
    dbl_strides = <size_t*> malloc(dims * sizeof(size_t))
    if vg_shape == NULL or dbl_strides == NULL:
      if vg_shape != NULL: free(vg_shape)
      if dbl_strides != NULL: free(dbl_strides)
      raise MemoryError('Allocation failure for voxel_graph strides')
    for a in range(dims):
      vg_shape[a] = <size_t>arr.shape[a]
    dbl_strides[dims-1] = 1
    for a in range(dims-2, -1, -1):
      dbl_strides[a] = dbl_strides[a+1] * (vg_shape[a+1] * 2)
    graph_data = np.PyArray_DATA(graph_u)
    fg_data = np.PyArray_DATA(fg)
    dbl_data = np.PyArray_DATA(dbl)
    with nogil:
      _gate_voxel_graph_uint64(<np.uint64_t*> graph_data,
                               <np.uint8_t*> fg_data,
                               <np.uint8_t*> dbl_data,
                               <size_t>nd, vg_shape, dbl_strides)
    free(vg_shape); free(dbl_strides)

    if black_border:
      for a in range(dims):
        slc = [slice(None)] * dims
        slc[a] = -1
        dbl[tuple(slc)] = 0

    # Recursive call on doubled grid (without voxel_graph) - uses barrier EDT
    anis2 = tuple(val / 2.0 for val in anis)
    transform2 = edtsq_nd(dbl, anis2, black_border, 1, None, order)
    slc = tuple(slice(0, None, 2) for _ in range(dims))
    return transform2[slc]

  # Standard EDT (no voxel_graph) - use barrier algorithm
  barrier_graph = _build_barrier_graph_from_labels(arr, parallel)

  # Handle black_border by padding with background (zeros)
  # This ensures array edges are treated as borders
  cdef bint was_padded = False
  if black_border:
    # Pad barrier_graph with zeros on all sides
    pad_width = [(1, 1)] * dims
    barrier_graph = np.pad(barrier_graph, pad_width, mode='constant', constant_values=0)
    was_padded = True

  if not barrier_graph.flags.c_contiguous:
    barrier_graph = np.ascontiguousarray(barrier_graph)

  # Prepare arrays for C++ call - use padded shape if applicable
  original_shape = tuple(arr.shape)
  # Compute working shape: if padded, add 2 to each dimension
  if was_padded:
    working_shape = tuple(s + 2 for s in original_shape)
  else:
    working_shape = original_shape
  shape_arr = np.array(working_shape, dtype=np.int64)
  anis_arr = np.array(anis, dtype=np.float32)
  cdef int64_t working_voxels = 1
  for s in working_shape:
    working_voxels *= s

  shape_ptr = <int64_t*> np.PyArray_DATA(shape_arr)
  anis_ptr = <float*> np.PyArray_DATA(anis_arr)
  barrier_ptr = np.PyArray_DATA(barrier_graph)

  # Call barrier EDT
  cdef bint is_u8 = barrier_graph.dtype == np.uint8
  cdef bint is_u16 = barrier_graph.dtype == np.uint16

  if is_u8:
    with nogil:
      result_ptr = edtsq_barrier[uint8_t](
        <const uint8_t*> barrier_ptr,
        shape_ptr,
        anis_ptr,
        ndim_c,
        parallel_c
      )
  elif is_u16:
    with nogil:
      result_ptr = edtsq_barrier[uint16_t](
        <const uint16_t*> barrier_ptr,
        shape_ptr,
        anis_ptr,
        ndim_c,
        parallel_c
      )
  else:
    raise TypeError("barrier_graph must be uint8 or uint16")

  # Copy result to numpy array and free C++ allocated memory
  result_full = np.empty(working_shape, dtype=np.float32)
  cdef float* result_flat = <float*> np.PyArray_DATA(result_full)
  memcpy(result_flat, result_ptr, working_voxels * sizeof(float))
  delete_array_barrier[float](result_ptr)

  # Extract center portion if we padded
  if was_padded:
    slc = tuple(slice(1, -1) for _ in range(dims))
    out = np.ascontiguousarray(result_full[slc])
  else:
    out = result_full

  if profile_enabled:
    profile_sections['total'] = time.perf_counter() - t_total_start
    profile_data['shape'] = original_shape
    profile_data['dims'] = dims
    profile_data['dtype'] = str(arr.dtype)
    profile_data['sections'] = profile_sections
    profile_data['axes'] = profile_axes
    _nd_profile_last = profile_data

  return out


@cython.binding(True)
def edtsq_nd_last_profile():
  """Return the last ND profile captured when EDT_ND_PROFILE is enabled."""
  return _nd_profile_last


@cython.binding(True)
def _debug_parabolic_multi(labels, values, int stride, anisotropy=1.0, black_border=False):
  """Internal helper for debugging parabolic pass on a single line."""
  arr = np.require(labels, dtype=np.uint8, requirements='C')
  vals = np.require(values, dtype=np.float32, requirements='C')
  cdef Py_ssize_t n = arr.size
  if vals.size != n:
    raise ValueError('values must match labels length')
  cdef int strd = max(1, stride)
  (<object>_parabolic_1d_typed)(arr, vals, <int>n, <long>strd, <float>anisotropy, black_border)
  return vals.copy()




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

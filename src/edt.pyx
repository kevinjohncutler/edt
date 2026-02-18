# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Multi-label Euclidean Distance Transform based on the algorithms of
Saito et al (1994), Meijster et al (2002), and Felzenszwalb & Huttenlocher (2012).

Uses connectivity graphs internally (uint8 for 2D-4D, uint16 for 5D+).
Memory-efficient for larger input dtypes (up to 38% savings for uint32 input
vs label-segment approaches).
Supports custom voxel_graph input for user-defined boundaries.

Key methods:
  edt, edtsq - main EDT functions
  edt_graph, edtsq_graph - EDT from pre-built connectivity graph
  build_graph - build connectivity graph from labels

Additional utilities (from edt_barrier):
  feature_transform, expand_labels, each, sdf

Programmatic configuration:
  edt.configure(...) - set threading parameters in-process (see configure docstring)

Environment Variables (runtime):
  EDT_ADAPTIVE_THREADS      - 0/1, enable adaptive thread limiting by array size (default: 1)
  EDT_ND_MIN_VOXELS_PER_THREAD - min voxels per thread for ND>=4 arrays (default: 50000)
  EDT_ND_MIN_LINES_PER_THREAD  - min lines per thread for ND>=4 arrays (default: 32)
  EDT_ND_PROFILE            - set to 1 to enable per-call profiling output

Environment Variables (build-time):
  EDT_MARCH_NATIVE          - 0/1, compile with -march=native (default: 1)

License: GNU 3.0

Original EDT: William Silversmith (Seung Lab, Princeton),  August 2018 - February 2026
ND connectivity graph EDT: Kevin Cutler, February 2026
"""

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from libc.stdlib cimport malloc, free
from libcpp cimport bool as native_bool
cimport numpy as np
np.import_array()

import numpy as np
import multiprocessing
import os


def _graph_dtype(ndim):
    """Return the minimal uint dtype for a connectivity graph of ndim dimensions.

    Each axis occupies 2 bits, so max bit = 2*(ndim-1):
      dims 1-4  -> uint8  (max bit 6)
      dims 5-8  -> uint16 (max bit 14)
      dims 9-12 -> uint32 (max bit 22)
      dims 13-16 -> uint64 (max bit 30)
    """
    if ndim > 16:
        raise ValueError(f"EDT supports at most 16 dimensions, got {ndim}.")
    return (np.uint8, np.uint16, np.uint32, np.uint64)[(ndim - 1) // 4]


cdef extern from "edt.hpp" namespace "nd":
    # Tuning
    cdef void _nd_set_tuning "nd::set_tuning"(size_t chunks_per_thread, size_t tile) nogil
    cdef void _nd_set_force_generic "nd::set_force_generic"(native_bool force) nogil

    # EDT from labels (original, unused)
    cdef void edtsq_from_labels[T](
        const T* labels,
        float* output,
        const size_t* shape,
        const float* anisotropy,
        size_t dims,
        native_bool black_border,
        int parallel
    ) nogil

    # EDT from voxel graph
    cdef void edtsq_from_graph[GRAPH_T](
        const GRAPH_T* graph,
        float* output,
        const size_t* shape,
        const float* anisotropy,
        size_t dims,
        native_bool black_border,
        int parallel
    ) nogil

    # Build connectivity graph from labels
    cdef void build_connectivity_graph[T, GRAPH_T](
        const T* labels,
        GRAPH_T* graph,
        const size_t* shape,
        size_t dims,
        int parallel
    ) nogil

    # Fused: build graph internally then run EDT (more efficient)
    cdef void edtsq_from_labels_fused[T](
        const T* labels,
        float* output,
        const size_t* shape,
        const float* anisotropy,
        size_t dims,
        native_bool black_border,
        int parallel
    ) nogil

    # Expand labels helpers
    cdef void _nd_expand_init_bases[INDEX](
        const uint8_t* seeds,
        float* dist,
        const size_t* bases,
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
        const size_t* bases,
        size_t num_lines,
        size_t n,
        size_t s,
        float anis,
        native_bool black_border,
        INDEX* feat,
        int parallel
    ) nogil

    cdef void _nd_expand_init_labels_bases(
        const uint8_t* seeds,
        float* dist,
        const size_t* bases,
        size_t num_lines,
        size_t n,
        size_t s,
        float anis,
        native_bool black_border,
        const uint32_t* labelsp,
        uint32_t* label_out,
        int parallel
    ) nogil

    cdef void _nd_expand_parabolic_labels_bases(
        float* dist,
        const size_t* bases,
        size_t num_lines,
        size_t n,
        size_t s,
        float anis,
        native_bool black_border,
        const uint32_t* label_in,
        uint32_t* label_out,
        int parallel
    ) nogil


def set_tuning(chunks_per_thread=1, tile=8):
    """Set tuning parameters for ND EDT."""
    _nd_set_tuning(chunks_per_thread, tile)


def set_force_generic(force):
    """Force use of generic ND path (for testing/benchmarking). Deprecated - no-op."""
    _nd_set_force_generic(force)


def _voxel_graph_to_nd(voxel_graph, labels=None):
    """
    Convert bidirectional voxel_graph to ND graph format.

    The voxel_graph format uses 2*ndim bits per voxel:
    - positive direction at bit (2*(ndim-1-axis))
    - negative direction at bit (2*(ndim-1-axis)+1)

    The ND format uses forward edges only + foreground marker:
    - Forward edge for axis a at bit (2*(ndim-1-a))
    - Bit 7 (0x80) marks foreground

    Since positive direction bits match ND edge bits exactly,
    we just mask out negative bits and add foreground marker.

    If labels is None, foreground is inferred from voxel_graph != 0
    (any voxel with connectivity is foreground).
    """
    ndim = voxel_graph.ndim
    if labels is not None and voxel_graph.shape != labels.shape:
        raise ValueError("voxel_graph shape must match labels")

    # Build mask for positive direction bits only
    pos_mask = 0
    for axis in range(ndim):
        pos_mask |= (1 << (2 * (ndim - 1 - axis)))

    # Extract positive direction bits (these match ND edge bits)
    # Use minimal dtype based on ndim (not input dtype) to avoid large intermediates
    mask_dtype = _graph_dtype(ndim)
    graph = voxel_graph.astype(mask_dtype, copy=False) & mask_dtype(pos_mask)
    # Keep mask_dtype - don't truncate to uint8 (5D+ needs higher bits)

    # Add foreground marker - infer from voxel_graph if no labels provided
    if labels is not None:
        graph[labels != 0] |= 0x80
    else:
        graph[voxel_graph != 0] |= 0x80

    return graph


def edtsq(labels=None, anisotropy=None, black_border=False, parallel=0, voxel_graph=None, order=None):
    """
    Compute squared Euclidean distance transform via graph-first architecture.

    Converts labels to a uint8 connectivity graph internally, then computes EDT.
    The graph is built and freed in C++ for maximum efficiency.

    Parameters
    ----------
    labels : ndarray or None
        Input label array. Non-zero values are foreground.
        Can be None if voxel_graph is provided (foreground inferred from connectivity).
    anisotropy : tuple or None
        Physical voxel size for each dimension. Default is isotropic (1, 1, ...).
    black_border : bool
        Treat image boundary as an object boundary.
    parallel : int
        Number of threads. 0 means auto-detect.
    voxel_graph : ndarray, optional
        Per-voxel bitfield describing allowed connections. Positive direction
        bits are extracted and used for EDT computation. If labels is None,
        foreground is inferred from voxel_graph != 0.
    order : ignored
        For backwards compatibility.

    Returns
    -------
    ndarray
        Squared Euclidean distance transform (float32).
    """
    # Handle voxel_graph input by converting to ND graph format
    if voxel_graph is not None:
        voxel_graph = np.ascontiguousarray(voxel_graph)
        if labels is not None:
            labels = np.asarray(labels)
        graph = _voxel_graph_to_nd(voxel_graph, labels)
        return edtsq_graph(graph, anisotropy, black_border, parallel)

    if labels is None:
        raise ValueError("labels is required when voxel_graph is not provided")

    # Preserve input dtype where possible to avoid copies
    labels = np.asarray(labels)
    dtype = labels.dtype
    if dtype == np.bool_:
        dtype = np.uint8
    elif dtype not in (np.uint8, np.uint16, np.uint32, np.uint64):
        dtype = np.uint32  # Fallback for signed/float types

    labels = np.ascontiguousarray(labels, dtype=dtype)
    cdef size_t nd = labels.ndim
    cdef tuple shape = labels.shape

    if anisotropy is None:
        anisotropy = tuple([1.0] * nd)
    elif not hasattr(anisotropy, '__len__'):
        anisotropy = (float(anisotropy),) * nd
    else:
        anisotropy = tuple(float(a) for a in anisotropy)

    if len(anisotropy) != nd:
        raise ValueError(f"anisotropy must have {nd} elements")

    if parallel <= 0:
        parallel = multiprocessing.cpu_count()

    global _nd_profile_last
    _nd_profile_last = {
        'shape': shape,
        'dims': nd,
        'parallel_used': parallel,
    }

    cdef size_t* cshape = <size_t*> malloc(nd * sizeof(size_t))
    cdef float* canis = <float*> malloc(nd * sizeof(float))
    if cshape == NULL or canis == NULL:
        if cshape != NULL:
            free(cshape)
        if canis != NULL:
            free(canis)
        raise MemoryError('Allocation failure')

    cdef size_t i
    for i in range(nd):
        cshape[i] = <size_t>shape[i]
        canis[i] = <float>anisotropy[i]

    cdef np.ndarray output = np.zeros(shape, dtype=np.float32)
    cdef float* outp = <float*> np.PyArray_DATA(output)
    cdef native_bool bb = black_border
    cdef int par = parallel

    # Dispatch based on dtype to avoid unnecessary copies
    cdef int dtype_code = 0  # 0=uint8, 1=uint16, 2=uint32, 3=uint64
    if dtype == np.uint16:
        dtype_code = 1
    elif dtype == np.uint32:
        dtype_code = 2
    elif dtype == np.uint64:
        dtype_code = 3

    cdef uint8_t* labelsp8
    cdef uint16_t* labelsp16
    cdef uint32_t* labelsp32
    cdef uint64_t* labelsp64

    try:
        if dtype_code == 0:
            labelsp8 = <uint8_t*> np.PyArray_DATA(labels)
            with nogil:
                edtsq_from_labels_fused[uint8_t](labelsp8, outp, cshape, canis, nd, bb, par)
        elif dtype_code == 1:
            labelsp16 = <uint16_t*> np.PyArray_DATA(labels)
            with nogil:
                edtsq_from_labels_fused[uint16_t](labelsp16, outp, cshape, canis, nd, bb, par)
        elif dtype_code == 2:
            labelsp32 = <uint32_t*> np.PyArray_DATA(labels)
            with nogil:
                edtsq_from_labels_fused[uint32_t](labelsp32, outp, cshape, canis, nd, bb, par)
        else:  # uint64
            labelsp64 = <uint64_t*> np.PyArray_DATA(labels)
            with nogil:
                edtsq_from_labels_fused[uint64_t](labelsp64, outp, cshape, canis, nd, bb, par)
    finally:
        free(cshape)
        free(canis)

    return output


def edt(labels=None, anisotropy=None, black_border=False, parallel=0, voxel_graph=None, order=None):
    """
    Compute Euclidean distance transform.

    Returns the square root of edtsq.
    """
    dt = edtsq(labels, anisotropy, black_border, parallel, voxel_graph, order)
    return np.sqrt(dt, out=dt)


def edtsq_graph(graph, anisotropy=None, black_border=False, parallel=0):
    """
    Compute squared EDT from a voxel connectivity graph.

    Parameters
    ----------
    graph : ndarray (uint8 for 2D-4D, uint16 for 5D-8D)
        Voxel connectivity graph. Each element encodes edge bits for each axis.
        For 2D: axis 0 -> bit 2, axis 1 -> bit 0
        For 3D: axis 0 -> bit 4, axis 1 -> bit 2, axis 2 -> bit 0
    anisotropy : tuple or None
        Physical voxel size for each dimension.
    black_border : bool
        Treat image boundary as an object boundary.
    parallel : int
        Number of threads.

    Returns
    -------
    ndarray
        Squared Euclidean distance transform (float32).
    """
    cdef size_t nd = graph.ndim
    cdef tuple shape = graph.shape

    graph_dtype = _graph_dtype(nd)
    graph = np.ascontiguousarray(graph, dtype=graph_dtype)

    if anisotropy is None:
        anisotropy = tuple([1.0] * nd)
    elif not hasattr(anisotropy, '__len__'):
        anisotropy = (float(anisotropy),) * nd
    else:
        anisotropy = tuple(float(a) for a in anisotropy)

    if len(anisotropy) != nd:
        raise ValueError(f"anisotropy must have {nd} elements")

    if parallel <= 0:
        parallel = multiprocessing.cpu_count()

    global _nd_profile_last
    _nd_profile_last = {
        'shape': shape,
        'dims': nd,
        'parallel_used': parallel,
    }

    cdef size_t total = 1
    cdef size_t* cshape = <size_t*> malloc(nd * sizeof(size_t))
    cdef float* canis = <float*> malloc(nd * sizeof(float))
    if cshape == NULL or canis == NULL:
        if cshape != NULL:
            free(cshape)
        if canis != NULL:
            free(canis)
        raise MemoryError('Allocation failure')

    cdef size_t i
    for i in range(nd):
        cshape[i] = <size_t>shape[i]
        canis[i] = <float>anisotropy[i]
        total *= cshape[i]

    cdef np.ndarray output = np.zeros(shape, dtype=np.float32)
    cdef float* outp = <float*> np.PyArray_DATA(output)

    cdef native_bool bb = black_border
    cdef int par = parallel

    # Get graph pointer before nogil (dispatch based on dtype)
    cdef void* graphp = np.PyArray_DATA(graph)

    try:
        if nd <= 4:
            with nogil:
                edtsq_from_graph[uint8_t](<uint8_t*>graphp, outp, cshape, canis, nd, bb, par)
        elif nd <= 8:
            with nogil:
                edtsq_from_graph[uint16_t](<uint16_t*>graphp, outp, cshape, canis, nd, bb, par)
        elif nd <= 12:
            with nogil:
                edtsq_from_graph[uint32_t](<uint32_t*>graphp, outp, cshape, canis, nd, bb, par)
        else:
            with nogil:
                edtsq_from_graph[uint64_t](<uint64_t*>graphp, outp, cshape, canis, nd, bb, par)
    finally:
        free(cshape)
        free(canis)

    return output


def edt_graph(graph, anisotropy=None, black_border=False, parallel=0):
    """
    Compute EDT from a voxel connectivity graph.

    Returns the square root of edtsq_graph.
    """
    dt = edtsq_graph(graph, anisotropy, black_border, parallel)
    return np.sqrt(dt, out=dt)


def build_graph(labels, parallel=0):
    """
    Build a connectivity graph from labels.

    Parameters
    ----------
    labels : ndarray
        Input label array.
    parallel : int
        Number of threads.

    Returns
    -------
    ndarray (uint8)
        Connectivity graph where each byte encodes edge bits.
    """
    # Preserve input dtype where possible to avoid copies
    labels = np.asarray(labels)
    dtype = labels.dtype
    if dtype == np.bool_:
        dtype = np.uint8
    elif dtype not in (np.uint8, np.uint16, np.uint32, np.uint64):
        dtype = np.uint32

    labels = np.ascontiguousarray(labels, dtype=dtype)
    cdef size_t nd = labels.ndim
    cdef tuple shape = labels.shape

    if parallel <= 0:
        parallel = multiprocessing.cpu_count()

    cdef size_t total = 1
    cdef size_t* cshape = <size_t*> malloc(nd * sizeof(size_t))
    if cshape == NULL:
        raise MemoryError('Allocation failure')

    cdef size_t i
    for i in range(nd):
        cshape[i] = <size_t>shape[i]
        total *= cshape[i]

    cdef np.ndarray graph = np.zeros(shape, dtype=np.uint8)
    cdef uint8_t* graphp = <uint8_t*> np.PyArray_DATA(graph)
    cdef int par = parallel

    # Dispatch based on dtype
    cdef int dtype_code = 0  # 0=uint8, 1=uint16, 2=uint32, 3=uint64
    if dtype == np.uint16:
        dtype_code = 1
    elif dtype == np.uint32:
        dtype_code = 2
    elif dtype == np.uint64:
        dtype_code = 3

    cdef uint8_t* labelsp8
    cdef uint16_t* labelsp16
    cdef uint32_t* labelsp32
    cdef uint64_t* labelsp64

    try:
        if dtype_code == 0:
            labelsp8 = <uint8_t*> np.PyArray_DATA(labels)
            with nogil:
                build_connectivity_graph[uint8_t, uint8_t](labelsp8, graphp, cshape, nd, par)
        elif dtype_code == 1:
            labelsp16 = <uint16_t*> np.PyArray_DATA(labels)
            with nogil:
                build_connectivity_graph[uint16_t, uint8_t](labelsp16, graphp, cshape, nd, par)
        elif dtype_code == 2:
            labelsp32 = <uint32_t*> np.PyArray_DATA(labels)
            with nogil:
                build_connectivity_graph[uint32_t, uint8_t](labelsp32, graphp, cshape, nd, par)
        else:  # uint64
            labelsp64 = <uint64_t*> np.PyArray_DATA(labels)
            with nogil:
                build_connectivity_graph[uint64_t, uint8_t](labelsp64, graphp, cshape, nd, par)
    finally:
        free(cshape)

    return graph


# Signed Distance Function - positive inside foreground, negative in background
def sdf(data, anisotropy=None, black_border=False, int parallel=1):
    """
    Compute the Signed Distance Function (SDF).

    Foreground pixels get positive distance (to nearest background).
    Background pixels get negative distance (to nearest foreground).

    Args:
        data: Input array (binary or labels, 0 = background)
        anisotropy: Pixel spacing per dimension
        black_border: Treat image edges as background
        parallel: Number of threads

    Returns:
        SDF as float32 array
    """
    dt = edt(data, anisotropy=anisotropy, black_border=black_border, parallel=parallel)
    dt -= edt(data == 0, anisotropy=anisotropy, black_border=black_border, parallel=parallel)
    return dt

def sdfsq(data, anisotropy=None, black_border=False, int parallel=1):
    """Squared SDF - same as sdf but with squared distances."""
    dt = edtsq(data, anisotropy=anisotropy, black_border=black_border, parallel=parallel)
    dt -= edtsq(data == 0, anisotropy=anisotropy, black_border=black_border, parallel=parallel)
    return dt

# Import utilities from edt_legacy for backwards compatibility
try:
    from edt_legacy import each, draw, erase
    import edt_legacy as legacy
except ImportError:
    legacy = None

# expand_labels and feature_transform - ported from barrier implementation
def expand_labels(data, anisotropy=None, int parallel=1, return_features=False):
    """Expand nonzero labels to zeros by nearest-neighbor in Euclidean metric (ND).

    Parameters
    ----------
    data : ndarray
        Input array; nonzero elements are seeds whose values are the labels.
    anisotropy : float or sequence of float, optional
        Per-axis voxel size (default 1.0 for all axes).
    parallel : int, optional
        Number of threads; if <= 0, uses cpu_count().
    return_features : bool, optional
        If True, also return the feature (nearest-seed linear index) array.

    Returns
    -------
    labels : ndarray, dtype=uint32
        Expanded labels, same shape as input.
    features : ndarray, optional
        If return_features=True, the nearest-seed linear indices.
    """
    cdef np.ndarray[np.uint32_t, ndim=1] out1
    cdef np.ndarray pos
    cdef np.ndarray mids
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t k = 0
    cdef Py_ssize_t s
    cdef np.ndarray feat_prev
    cdef uint32_t* fprev_u32
    cdef size_t* fprev_sz
    cdef np.ndarray[np.uint32_t, ndim=1] out_flat
    cdef np.uint32_t* outp
    cdef np.uint32_t* labp2
    cdef size_t idx2
    cdef Py_ssize_t total_voxels
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

    # General ND implementation (dims >= 2)
    cdef native_bool bb = False
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    cdef np.ndarray[np.uint8_t, ndim=1] seeds_flat = (arr.ravel(order='K') != 0).astype(np.uint8, order='C')
    cdef np.ndarray[np.uint32_t, ndim=1] labels_flat = arr.ravel(order='K').astype(np.uint32, order='C')
    cdef tuple shape = arr.shape
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
    cstrides[nd-1] = 1
    for ii in range(nd-2, -1, -1):
        cstrides[ii] = cstrides[ii+1] * cshape[ii+1]
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

    cdef size_t total = 1
    for ii in range(nd): total *= cshape[ii]
    cdef np.ndarray[np.float32_t, ndim=1] dist = np.empty((total,), dtype=np.float32)
    cdef float* distp = <float*> np.PyArray_DATA(dist)
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

    cdef Py_ssize_t ax0 = paxes[0]
    cdef size_t n0 = cshape[ax0]
    cdef size_t s0 = cstrides[ax0]
    cdef size_t lines = total // n0
    cdef size_t max_lines = lines
    for ii in range(1, nd):
        if total // cshape[paxes[ii]] > max_lines:
            max_lines = total // cshape[paxes[ii]]
    cdef size_t* bases = <size_t*> malloc(max_lines * sizeof(size_t))
    if bases == NULL:
        free(cshape); free(cstrides); free(paxes); free(canis)
        raise MemoryError('Allocation failure for bases')
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

    for a in range(1, nd):
        lines = total // cshape[paxes[a]]
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
        with nogil:
            for il in range(lines):
                base_b = 0
                tmp_b = il
                for j in range(ord_len):
                    coord_b = tmp_b % cshape[ord[j]]
                    base_b += coord_b * cstrides[ord[j]]
                    tmp_b //= cshape[ord[j]]
                bases[il] = base_b
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
            tmpL = lab_prev; lab_prev = lab_next; lab_next = tmpL
            lprev = <uint32_t*> np.PyArray_DATA(lab_prev)
            lnext = <uint32_t*> np.PyArray_DATA(lab_next)

    free(bases); free(ord); free(cshape); free(cstrides); free(paxes); free(canis)
    if return_features:
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


# ND aliases - our main edt/edtsq already use unified ND path
def edt_nd(data, anisotropy=None, black_border=False, int parallel=1, voxel_graph=None, order=None):
    """ND EDT - alias for edt() which uses unified ND implementation."""
    return edt(data, anisotropy=anisotropy, black_border=black_border, parallel=parallel, voxel_graph=voxel_graph, order=order)

def edtsq_nd(data, anisotropy=None, black_border=False, int parallel=1, voxel_graph=None, order=None):
    """ND squared EDT - alias for edtsq() which uses unified ND implementation."""
    return edtsq(data, anisotropy=anisotropy, black_border=black_border, parallel=parallel, voxel_graph=voxel_graph, order=order)

_nd_profile_last = None
def edtsq_nd_last_profile():
    """Return the last ND profile."""
    return _nd_profile_last

# Thread limiting heuristics
_ND_2D_THRESHOLDS = [100000, 500000, 1000000]
_ND_2D_MAX_THREADS = [1, 2, 4]
_ND_3D_THRESHOLDS = [64, 128, 256]
_ND_3D_MAX_THREADS = [1, 2, 4]
_ND_MIN_VOXELS_PER_THREAD_DEFAULT = 50000
_ND_MIN_LINES_PER_THREAD_DEFAULT = 32

# In-process overrides set via configure(), take priority over env vars
_ND_CONFIG = {}

def configure(
    adaptive_threads=None,
    min_voxels_per_thread=None,
    min_lines_per_thread=None,
):
    """
    Set EDT threading parameters programmatically, overriding environment
    variables for the current process.

    Parameters
    ----------
    adaptive_threads : bool or None
        Enable adaptive thread limiting based on array size.
        Overrides EDT_ADAPTIVE_THREADS.
    min_voxels_per_thread : int or None
        Minimum voxels per thread for ND>=4 arrays.
        Overrides EDT_ND_MIN_VOXELS_PER_THREAD.
    min_lines_per_thread : int or None
        Minimum lines per thread for ND>=4 arrays.
        Overrides EDT_ND_MIN_LINES_PER_THREAD.
    """
    if adaptive_threads is not None:
        _ND_CONFIG['EDT_ADAPTIVE_THREADS'] = int(bool(adaptive_threads))
    if min_voxels_per_thread is not None:
        _ND_CONFIG['EDT_ND_MIN_VOXELS_PER_THREAD'] = int(min_voxels_per_thread)
    if min_lines_per_thread is not None:
        _ND_CONFIG['EDT_ND_MIN_LINES_PER_THREAD'] = int(min_lines_per_thread)

def _env_int(name, default):
    if name in _ND_CONFIG:
        return _ND_CONFIG[name]
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default

def _apply_thresholds(val, thresholds, max_threads, capped):
    for thresh, maxT in zip(thresholds, max_threads):
        if val < thresh:
            return min(capped, maxT)
    return capped

def _adaptive_thread_limit_nd(parallel, shape, requested=None):
    """General ND thread limiter matching 2D/3D heuristics."""
    parallel = max(1, parallel)
    adapt_enabled = bool(_env_int('EDT_ADAPTIVE_THREADS', 1))
    if not adapt_enabled:
        return parallel
    dims = len(shape)
    if dims <= 1:
        return parallel
    capped = parallel
    if dims == 2:
        total = int(shape[0] * shape[1])
        capped = _apply_thresholds(total, _ND_2D_THRESHOLDS, _ND_2D_MAX_THREADS, capped)
        return max(1, capped)
    if dims == 3:
        longest = int(max(shape))
        capped = _apply_thresholds(longest, _ND_3D_THRESHOLDS, _ND_3D_MAX_THREADS, capped)
        return max(1, capped)
    min_voxels = _env_int('EDT_ND_MIN_VOXELS_PER_THREAD', _ND_MIN_VOXELS_PER_THREAD_DEFAULT)
    if min_voxels < 1:
        min_voxels = _ND_MIN_VOXELS_PER_THREAD_DEFAULT
    min_lines = _env_int('EDT_ND_MIN_LINES_PER_THREAD', _ND_MIN_LINES_PER_THREAD_DEFAULT)
    if min_lines < 1:
        min_lines = _ND_MIN_LINES_PER_THREAD_DEFAULT
    total = 1
    for extent in shape:
        total *= extent
    longest = max(shape)
    lines = max(1, total // longest)
    cap_lines = max(1, lines // min_lines)
    cap_voxels = max(1, total // min_voxels)
    cap = max(cap_lines, cap_voxels)
    capped = min(capped, cap)
    return max(1, capped)


def feature_transform(data, anisotropy=None, black_border=False, int parallel=1, return_distances=False, features_dtype='auto'):
    """ND feature transform (nearest seed) with optional Euclidean distances.

    Parameters
    ----------
    data : ndarray
        Seed image (nonzero are seeds).
    anisotropy : float or sequence of float, optional
        Per-axis voxel size (default 1.0 for all axes).
    black_border : bool, optional
        If True, treat the border as background (default False).
    parallel : int, optional
        Number of threads; if <= 0, uses cpu_count().
    return_distances : bool, optional
        If True, also return the EDT of the seed mask.
    features_dtype : dtype-like, optional
        Output dtype for the feature index array. Accepts any value accepted
        by np.dtype() (e.g. np.uint32, 'uint64'). Defaults to np.uint32 for
        arrays with fewer than 2**32 voxels, np.uint64 otherwise.

    Returns
    -------
    feat : ndarray
        Linear index of nearest seed for each voxel.
    dist : ndarray of float32, optional
        Euclidean distance to nearest seed, if return_distances=True.
    """
    arr = np.asarray(data)
    if arr.size == 0:
        if return_distances:
            return np.zeros_like(arr, dtype=np.uint64), np.zeros_like(arr, dtype=np.float32)
        return np.zeros_like(arr, dtype=np.uint64)

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

    labels, feats = expand_labels(arr.astype(np.uint32, copy=False), anis, parallel, True)

    voxels = arr.size
    if features_dtype == 'auto' or features_dtype is None:
        out_dtype = np.uint32 if voxels < 2**32 else np.uint64
    else:
        out_dtype = np.dtype(features_dtype)
        if not np.issubdtype(out_dtype, np.unsignedinteger):
            raise ValueError(f"features_dtype must be an unsigned integer dtype, got {out_dtype}")
    feats = feats.astype(out_dtype, copy=False)

    if return_distances:
        dist = edt((arr != 0).astype(np.uint8, copy=False), anis, black_border, parallel)
        return feats, dist
    return feats

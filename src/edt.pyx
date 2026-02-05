# cython: language_level=3
"""
EDT - Euclidean Distance Transform

Multi-label EDT using graph-first architecture. Builds a connectivity graph
from labels, then computes EDT using a unified ND algorithm.

33% less memory than segment-label approach (uint8 graph vs uint32 labels).
"""
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from libc.stdlib cimport malloc, free
from libcpp cimport bool as native_bool
cimport numpy as np
np.import_array()

import numpy as np
import multiprocessing


cdef extern from "nd_v2_core.hpp" namespace "nd_v2":
    # Tuning
    cdef void v2_set_tuning(size_t chunks_per_thread, size_t tile) nogil
    cdef void v2_set_force_generic(native_bool force) nogil

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


def set_tuning(chunks_per_thread=1, tile=8):
    """Set tuning parameters for nd_v2 EDT."""
    v2_set_tuning(chunks_per_thread, tile)


def set_force_generic(force):
    """Force use of generic ND path (for testing/benchmarking)."""
    v2_set_force_generic(force)


def _voxel_graph_to_nd_v2(labels, voxel_graph):
    """
    Convert bidirectional voxel_graph to nd_v2 graph format.

    The voxel_graph format uses 2*ndim bits per voxel:
    - positive direction at bit (2*(ndim-1-axis))
    - negative direction at bit (2*(ndim-1-axis)+1)

    The nd_v2 format uses forward edges only + foreground marker:
    - Forward edge for axis a at bit (2*(ndim-1-a))
    - Bit 7 (0x80) marks foreground

    Since positive direction bits match nd_v2 edge bits exactly,
    we just mask out negative bits and add foreground marker.
    """
    ndim = voxel_graph.ndim
    if voxel_graph.shape != labels.shape:
        raise ValueError("voxel_graph shape must match labels")

    # Build mask for positive direction bits only
    pos_mask = 0
    for axis in range(ndim):
        pos_mask |= (1 << (2 * (ndim - 1 - axis)))

    # Extract positive direction bits (these match nd_v2 edge bits)
    graph = (voxel_graph.astype(np.uint32) & pos_mask).astype(np.uint8)

    # Add foreground marker for foreground voxels
    graph[labels != 0] |= 0x80

    return graph


def edtsq(labels, anisotropy=None, black_border=False, parallel=0, voxel_graph=None, order=None):
    """
    Compute squared Euclidean distance transform via graph-first architecture.

    Converts labels to a uint8 connectivity graph internally, then computes EDT.
    The graph is built and freed in C++ for maximum efficiency.

    Parameters
    ----------
    labels : ndarray
        Input label array. Non-zero values are foreground.
    anisotropy : tuple or None
        Physical voxel size for each dimension. Default is isotropic (1, 1, ...).
    black_border : bool
        Treat image boundary as an object boundary.
    parallel : int
        Number of threads. 0 means auto-detect.
    voxel_graph : ndarray, optional
        Per-voxel bitfield describing allowed connections. Positive direction
        bits are extracted and used for EDT computation.
    order : ignored
        For backwards compatibility.

    Returns
    -------
    ndarray
        Squared Euclidean distance transform (float32).
    """
    # Handle voxel_graph input by converting to nd_v2 format
    if voxel_graph is not None:
        labels = np.asarray(labels)
        graph = _voxel_graph_to_nd_v2(labels, np.ascontiguousarray(voxel_graph))
        return edtsq_graph(graph, anisotropy, black_border, parallel)

    labels = np.ascontiguousarray(labels, dtype=np.uint32)
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
    cdef uint32_t* labelsp = <uint32_t*> np.PyArray_DATA(labels)

    cdef native_bool bb = black_border
    cdef int par = parallel

    try:
        with nogil:
            edtsq_from_labels_fused[uint32_t](labelsp, outp, cshape, canis, nd, bb, par)
    finally:
        free(cshape)
        free(canis)

    return output


def edt(labels, anisotropy=None, black_border=False, parallel=0, voxel_graph=None, order=None):
    """
    Compute Euclidean distance transform.

    Returns the square root of edtsq.
    """
    return np.sqrt(edtsq(labels, anisotropy, black_border, parallel, voxel_graph, order))


def edtsq_graph(graph, anisotropy=None, black_border=False, parallel=0):
    """
    Compute squared EDT from a voxel connectivity graph.

    Parameters
    ----------
    graph : ndarray (uint8)
        Voxel connectivity graph. Each byte encodes edge bits for each axis.
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
    graph = np.ascontiguousarray(graph, dtype=np.uint8)
    cdef size_t nd = graph.ndim
    cdef tuple shape = graph.shape

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
    cdef uint8_t* graphp = <uint8_t*> np.PyArray_DATA(graph)

    cdef native_bool bb = black_border
    cdef int par = parallel

    try:
        with nogil:
            edtsq_from_graph[uint8_t](graphp, outp, cshape, canis, nd, bb, par)
    finally:
        free(cshape)
        free(canis)

    return output


def edt_graph(graph, anisotropy=None, black_border=False, parallel=0):
    """
    Compute EDT from a voxel connectivity graph.

    Returns the square root of edtsq_graph.
    """
    return np.sqrt(edtsq_graph(graph, anisotropy, black_border, parallel))


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
    labels = np.ascontiguousarray(labels, dtype=np.uint32)
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
    cdef uint32_t* labelsp = <uint32_t*> np.PyArray_DATA(labels)
    cdef uint8_t* graphp = <uint8_t*> np.PyArray_DATA(graph)
    cdef int par = parallel

    try:
        with nogil:
            build_connectivity_graph[uint32_t, uint8_t](labelsp, graphp, cshape, nd, par)
    finally:
        free(cshape)

    return graph


# Import additional functions from edt_barrier for backwards compatibility
# These are utility functions that don't need the optimized graph-first path
try:
    from edt_barrier import (
        expand_labels,
        feature_transform,
        each,
        # ND-specific functions
        edt_nd,
        edtsq_nd,
        edtsq_nd_last_profile,
        _adaptive_thread_limit_nd,
        # Other utilities
        sdf,
        draw,
        erase,
        # Legacy module reference
        legacy,
        # Thread limit heuristics
        nd_tuning,
    )
except ImportError:
    pass  # edt_barrier not available

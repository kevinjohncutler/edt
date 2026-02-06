# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Multi-label Euclidean Distance Transform based on the algorithms of
Saito et al (1994), Meijster et al (2002), and Felzenszwalb & Huttenlocher (2012).

Uses uint8 connectivity graphs for 25% memory reduction vs segment labels.
Supports custom voxel_graph input for user-defined boundaries.

Key methods:
  edt, edtsq - main EDT functions
  edt_graph, edtsq_graph - EDT from pre-built connectivity graph
  build_graph - build connectivity graph from labels

Additional utilities (from edt_barrier):
  feature_transform, expand_labels, each, sdf

License: GNU 3.0

Original EDT: William Silversmith (Seung Lab, Princeton), 2018-2023
ND connectivity graph EDT: Kevin Cutler, 2026
"""
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from libc.stdlib cimport malloc, free
from libcpp cimport bool as native_bool
cimport numpy as np
np.import_array()

import numpy as np
import multiprocessing


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
    graph = (voxel_graph.astype(np.uint32) & pos_mask).astype(np.uint8)

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


def edt(labels=None, anisotropy=None, black_border=False, parallel=0, voxel_graph=None, order=None):
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

#!/usr/bin/env python3
"""
Test correctness of ND EDT implementation.
"""
import numpy as np
import pytest
import sys
import os
import multiprocessing

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from debug_utils import make_label_matrix
import edt

def _make_bench_array(shape, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 3, size=shape, dtype=np.uint8)
    if arr.ndim == 2:
        y, x = shape
        if y > 20 and x > 20:
            arr[y // 4 : y // 2, x // 4 : x // 2] = 1
            arr[3 * y // 5 : 4 * y // 5, 3 * x // 5 : 4 * x // 5] = 2
    elif arr.ndim == 3:
        z, y, x = shape
        if z > 10 and y > 20 and x > 20:
            arr[z // 4 : z // 3, y // 4 : y // 2, x // 4 : x // 2] = 1
            arr[3 * z // 5 : 4 * z // 5, 3 * y // 5 : 4 * y // 5, 3 * x // 5 : 4 * x // 5] = 2
    return arr

def test_nd_correctness_2d():
    """Test ND EDT correctness for 2D cases."""
    for M in [50, 100, 200]:
        masks = make_label_matrix(2, M)
        r1 = edt.edt(masks, parallel=-1)
        r2 = edt.edt_nd(masks, parallel=-1)
        np.testing.assert_allclose(r1, r2, rtol=1e-6, atol=1e-6,
                                   err_msg=f"2D case M={M} failed")

        expected_max = float(M)
        np.testing.assert_allclose(r1.max(), expected_max, rtol=1e-6, atol=1e-6 * expected_max,
                                   err_msg=f"2D edt max mismatch for M={M}")
        np.testing.assert_allclose(r2.max(), expected_max, rtol=1e-6, atol=1e-6 * expected_max,
                                   err_msg=f"2D edt_nd max mismatch for M={M}")

def test_nd_correctness_3d():
    """Test ND EDT correctness for 3D cases."""
    for M in [50, 100, 200]:
        masks = make_label_matrix(3, M)
        r1 = edt.edt(masks, parallel=-1)
        r2 = edt.edt_nd(masks, parallel=-1)
        np.testing.assert_allclose(r1, r2, rtol=1e-6, atol=1e-6,
                                   err_msg=f"3D case M={M} failed")

        expected_max = float(M)
        np.testing.assert_allclose(r1.max(), expected_max, rtol=1e-6, atol=1e-6 * expected_max,
                                   err_msg=f"3D edt max mismatch for M={M}")
        np.testing.assert_allclose(r2.max(), expected_max, rtol=1e-6, atol=1e-6 * expected_max,
                                   err_msg=f"3D edt_nd max mismatch for M={M}")

def test_nd_correctness_4d():
    """Test ND EDT correctness for 4D case (ND only, original doesn't support 4D)."""
    # Smaller size for 4D to keep test fast
    masks = make_label_matrix(4, 20)
    # Only test that ND doesn't crash on 4D
    r2 = edt.edt_nd(masks, parallel=-1)
    assert r2.shape == masks.shape, "4D ND EDT shape mismatch"
    assert np.all(np.isfinite(r2)), "4D ND EDT produced non-finite values"

    expected_max = float(20)
    np.testing.assert_allclose(r2.max(), expected_max, rtol=1e-6, atol=1e-6 * expected_max,
                               err_msg="4D edt_nd max mismatch")


def test_nd_correctness_5d():
    """Test ND EDT correctness for 5D case (ND only)."""
    masks = make_label_matrix(5, 10)
    r2 = edt.edt_nd(masks, parallel=-1)

    assert r2.shape == masks.shape, "5D ND EDT shape mismatch"
    assert np.all(np.isfinite(r2)), "5D ND EDT produced non-finite values"

    expected_max = float(10)
    np.testing.assert_allclose(r2.max(), expected_max, rtol=1e-6, atol=1e-6 * expected_max,
                               err_msg="5D edt_nd max mismatch")

def test_nd_threading_consistency():
    """Test that threading produces consistent results."""
    masks = make_label_matrix(3, 50)
    
    # Compare serial vs threaded
    r_serial = edt.edt_nd(masks, parallel=1)
    r_threaded = edt.edt_nd(masks, parallel=-1)

    np.testing.assert_allclose(r_serial, r_threaded, rtol=1e-6, atol=1e-6,
                               err_msg="Threading consistency failed")


def _profile_parallel_used(arr, parallel):
    os.environ['EDT_ND_PROFILE'] = '1'
    try:
        edt.edtsq_nd(arr, parallel=parallel)
        profile = edt.edtsq_nd_last_profile()
    finally:
        os.environ.pop('EDT_ND_PROFILE', None)
    assert profile is not None, "Expected ND profile to be available"
    assert profile.get('parallel_requested') == parallel, (
        "Profile should record requested parallel"
    )
    used = profile.get('parallel_used')
    assert used is not None, "Profile missing parallel_used"
    return int(used)


def _expected_parallel_used(shape, requested):
    cpu_cap = multiprocessing.cpu_count()
    parallel = requested
    if parallel <= 0:
        parallel = cpu_cap
    else:
        parallel = max(1, min(parallel, cpu_cap))
    return edt._adaptive_thread_limit_nd(parallel, shape, requested)


def test_nd_thread_limit_heuristics():
    """Verify heuristic caps reduce oversubscription across shapes."""
    arr_128 = np.zeros((128, 128), dtype=np.uint8)
    assert _profile_parallel_used(arr_128, 16) == _expected_parallel_used(arr_128.shape, 16)
    assert _profile_parallel_used(arr_128, -1) == _expected_parallel_used(arr_128.shape, -1)

    arr_512 = np.zeros((512, 512), dtype=np.uint8)
    assert _profile_parallel_used(arr_512, 16) == _expected_parallel_used(arr_512.shape, 16)

    arr_192 = np.zeros((192, 192, 192), dtype=np.uint8)
    assert _profile_parallel_used(arr_192, -1) == _expected_parallel_used(arr_192.shape, -1)
    assert _profile_parallel_used(arr_192, 32) == _expected_parallel_used(arr_192.shape, 32)

@pytest.mark.parametrize(
    "shape",
    [
        (96, 96),
        (128, 128),
        (48, 48, 48),
        (64, 64, 64),
    ],
)
def test_nd_random_label_bench_patterns(shape):
    """Ensure ND path matches specialized kernels on benchmark-style random labels."""
    arr = _make_bench_array(shape, seed=0)
    assert arr.ndim in (2, 3), "Benchmark patterns currently cover 2D/3D cases"

    for parallel in (1, 4):
        spec = edt.edtsq(arr, parallel=parallel)
        nd = edt.edtsq_nd(arr, parallel=parallel)

        assert np.all(np.isfinite(spec)), "Specialized EDT produced non-finite values"
        assert np.all(np.isfinite(nd)), "ND EDT produced non-finite values"

        np.testing.assert_allclose(
            spec, nd, rtol=1e-6, atol=1e-6,
            err_msg=f"Random benchmark array mismatch for shape={shape} parallel={parallel}"
        )

if __name__ == "__main__":
    test_nd_correctness_2d()
    print("2D tests passed!")
    test_nd_correctness_3d()
    print("3D tests passed!")
    test_nd_correctness_4d()
    print("4D tests passed!")
    test_nd_threading_consistency()
    print("Threading tests passed!")
    print("All tests passed!")

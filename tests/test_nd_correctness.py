#!/usr/bin/env python3
"""
Test correctness of ND EDT implementation.
"""
import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from debug_utils import make_label_matrix
import edt

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

"""Tests for Fortran-order array support in edt."""

import numpy as np
import pytest
import edt


@pytest.fixture
def labels_3d():
    rng = np.random.default_rng(42)
    return (rng.random((20, 30, 40)) > 0.1).astype(np.uint8)


def test_fortran_matches_c_order_2d():
    labels = np.ones((32, 48), dtype=np.uint8)
    labels[10:20, 10:20] = 0
    labels_f = np.asfortranarray(labels)
    assert labels_f.flags.f_contiguous

    result_c = edt.edt(labels)
    result_f = edt.edt(labels_f)
    np.testing.assert_allclose(result_c, result_f, rtol=1e-5)


def test_fortran_matches_c_order_3d(labels_3d):
    labels_f = np.asfortranarray(labels_3d)
    assert labels_f.flags.f_contiguous

    result_c = edt.edt(labels_3d)
    result_f = edt.edt(labels_f)
    np.testing.assert_allclose(result_c, result_f, rtol=1e-5)


def test_fortran_output_is_fortran_contiguous(labels_3d):
    labels_f = np.asfortranarray(labels_3d)
    result = edt.edt(labels_f)
    assert result.flags.f_contiguous, "Output should be F-contiguous for F-contiguous input"
    assert result.shape == labels_3d.shape


def test_fortran_no_copy_for_correct_dtype():
    """F-contiguous uint8 input should not be copied (output shares no buffer with input,
    but the *input* itself should not be needlessly copied to C-order)."""
    labels = np.asfortranarray(np.ones((20, 20), dtype=np.uint8))
    labels[5:15, 5:15] = 0
    arr, is_f = edt._prepare_array(labels, np.uint8)
    assert is_f, "_prepare_array should detect F-contiguous input"
    assert arr.flags.f_contiguous, "_prepare_array should preserve F-order"


def test_fortran_with_anisotropy(labels_3d):
    anis = (2.0, 1.0, 0.5)
    labels_f = np.asfortranarray(labels_3d)

    result_c = edt.edt(labels_3d, anisotropy=anis)
    result_f = edt.edt(labels_f, anisotropy=anis)
    np.testing.assert_allclose(result_c, result_f, rtol=1e-5)


def test_1d_unaffected_by_fortran_path():
    """1D arrays are both C and F contiguous — should take C path."""
    labels = np.array([1, 1, 0, 1, 1], dtype=np.uint8)
    assert labels.flags.c_contiguous and labels.flags.f_contiguous
    arr, is_f = edt._prepare_array(labels, np.uint8)
    assert not is_f, "1D array should take C path even though f_contiguous is True"


def test_non_contiguous_falls_back_to_c():
    labels = np.ones((20, 20), dtype=np.uint8)
    sliced = labels[::2, ::2]  # Non-contiguous slice
    assert not sliced.flags.c_contiguous and not sliced.flags.f_contiguous
    arr, is_f = edt._prepare_array(sliced, np.uint8)
    assert not is_f
    assert arr.flags.c_contiguous

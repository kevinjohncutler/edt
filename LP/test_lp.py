"""Tests for edt_lp: general Lp distance transform."""
import numpy as np
import pytest
import edt
import edt_lp


# ---------------------------------------------------------------------------
# Brute-force reference implementation
# ---------------------------------------------------------------------------

def _brute_force_lp(labels, p, anisotropy=None, black_border=False):
    """O(n^2) reference Lp^p distance transform for small arrays."""
    labels = np.asarray(labels)
    ndim = labels.ndim
    shape = labels.shape
    if anisotropy is None:
        anisotropy = (1.0,) * ndim
    out = np.zeros(shape, dtype=np.float64)
    for idx in np.ndindex(*shape):
        if labels[idx] == 0:
            continue
        best = float("inf")
        my_label = labels[idx]
        for jdx in np.ndindex(*shape):
            if labels[jdx] != my_label or labels[jdx] == 0:
                dist_p = sum(
                    abs(anisotropy[d] * (idx[d] - jdx[d])) ** p
                    for d in range(ndim)
                )
                best = min(best, dist_p)
        if black_border:
            for d in range(ndim):
                best = min(best, abs(anisotropy[d] * (idx[d] + 1)) ** p)
                best = min(best, abs(anisotropy[d] * (shape[d] - idx[d])) ** p)
        out[idx] = best
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# p=2 parity with edt.edtsq (must match exactly)
# ---------------------------------------------------------------------------

class TestP2Parity:
    def test_1d(self):
        labels = np.array([0, 1, 1, 1, 0], dtype=np.uint8)
        ref = edt.edtsq(labels, black_border=True)
        got = edt_lp.edtp(labels, p=2.0, black_border=True)
        np.testing.assert_array_equal(ref, got)

    def test_1d_no_border(self):
        labels = np.array([0, 1, 1, 1, 1, 1, 0], dtype=np.uint8)
        ref = edt.edtsq(labels)
        got = edt_lp.edtp(labels, p=2.0)
        np.testing.assert_array_equal(ref, got)

    def test_2d(self):
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 4, size=(200, 200), dtype=np.uint8)
        ref = edt.edtsq(labels, parallel=1)
        got = edt_lp.edtp(labels, p=2.0, parallel=1)
        np.testing.assert_array_equal(ref, got)

    def test_3d(self):
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 3, size=(50, 50, 50), dtype=np.uint8)
        ref = edt.edtsq(labels, parallel=1)
        got = edt_lp.edtp(labels, p=2.0, parallel=1)
        np.testing.assert_array_equal(ref, got)

    def test_2d_parallel(self):
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 4, size=(200, 200), dtype=np.uint8)
        ref = edt.edtsq(labels, parallel=4)
        got = edt_lp.edtp(labels, p=2.0, parallel=4)
        np.testing.assert_array_equal(ref, got)

    def test_anisotropy(self):
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 3, size=(100, 100), dtype=np.uint8)
        anis = (1.5, 0.8)
        ref = edt.edtsq(labels, anisotropy=anis, parallel=1)
        got = edt_lp.edtp(labels, p=2.0, anisotropy=anis, parallel=1)
        np.testing.assert_array_equal(ref, got)

    def test_black_border(self):
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 3, size=(100, 100), dtype=np.uint8)
        ref = edt.edtsq(labels, black_border=True, parallel=1)
        got = edt_lp.edtp(labels, p=2.0, black_border=True, parallel=1)
        np.testing.assert_array_equal(ref, got)

    def test_uint16_labels(self):
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 300, size=(100, 100), dtype=np.uint16)
        ref = edt.edtsq(labels, parallel=1)
        got = edt_lp.edtp(labels, p=2.0, parallel=1)
        np.testing.assert_array_equal(ref, got)

    def test_uint32_labels(self):
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 300, size=(50, 50), dtype=np.uint32)
        ref = edt.edtsq(labels, parallel=1)
        got = edt_lp.edtp(labels, p=2.0, parallel=1)
        np.testing.assert_array_equal(ref, got)


# ---------------------------------------------------------------------------
# Brute-force validation for various p values
# ---------------------------------------------------------------------------

class TestBruteForce:
    @pytest.mark.parametrize("p", [1.0, 1.5, 2.0, 3.0, 4.0])
    def test_1d_binary(self, p):
        labels = np.array([0, 1, 1, 1, 1, 1, 0], dtype=np.uint8)
        got = edt_lp.edtp(labels, p=p, black_border=True)
        ref = _brute_force_lp(labels, p, black_border=True)
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("p", [1.0, 1.5, 2.0, 3.0, 4.0])
    def test_1d_multilabel(self, p):
        labels = np.array([1, 1, 1, 0, 2, 2, 0, 3, 3, 3, 3, 0], dtype=np.uint8)
        got = edt_lp.edtp(labels, p=p, black_border=True)
        ref = _brute_force_lp(labels, p, black_border=True)
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("p", [1.0, 1.5, 2.0, 3.0, 4.0])
    def test_1d_no_border(self, p):
        labels = np.array([0, 1, 1, 1, 1, 1, 0, 2, 2, 0], dtype=np.uint8)
        got = edt_lp.edtp(labels, p=p, black_border=False)
        ref = _brute_force_lp(labels, p, black_border=False)
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("p", [1.0, 1.5, 2.0, 3.0, 4.0])
    @pytest.mark.parametrize("seed", [123, 456, 789])
    def test_2d_small(self, p, seed):
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, 4, size=(10, 12), dtype=np.uint8)
        got = edt_lp.edtp(labels, p=p, black_border=True, parallel=1)
        ref = _brute_force_lp(labels, p, black_border=True)
        np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("p", [1.0, 1.5, 2.0, 3.0, 4.0])
    def test_2d_anisotropy(self, p):
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 3, size=(12, 12), dtype=np.uint8)
        anis = (1.5, 0.8)
        got = edt_lp.edtp(labels, p=p, anisotropy=anis, black_border=True, parallel=1)
        ref = _brute_force_lp(labels, p, anisotropy=anis, black_border=True)
        np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("p", [1.0, 1.5, 2.0, 3.0, 4.0])
    def test_2d_no_border(self, p):
        rng = np.random.default_rng(77)
        labels = rng.integers(0, 3, size=(10, 10), dtype=np.uint8)
        got = edt_lp.edtp(labels, p=p, black_border=False, parallel=1)
        ref = _brute_force_lp(labels, p, black_border=False)
        np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("p", [1.0, 1.5, 2.0, 3.0, 4.0])
    def test_3d_small(self, p):
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 2, size=(8, 8, 8), dtype=np.uint8)
        got = edt_lp.edtp(labels, p=p, black_border=True, parallel=1)
        ref = _brute_force_lp(labels, p, black_border=True)
        np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
    def test_3d_anisotropy(self, p):
        rng = np.random.default_rng(55)
        labels = rng.integers(0, 2, size=(7, 8, 6), dtype=np.uint8)
        anis = (2.0, 1.0, 0.5)
        got = edt_lp.edtp(labels, p=p, anisotropy=anis, black_border=True, parallel=1)
        ref = _brute_force_lp(labels, p, anisotropy=anis, black_border=True)
        np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
    def test_3d_no_border(self, p):
        rng = np.random.default_rng(33)
        labels = rng.integers(0, 3, size=(7, 7, 7), dtype=np.uint8)
        got = edt_lp.edtp(labels, p=p, black_border=False, parallel=1)
        ref = _brute_force_lp(labels, p, black_border=False)
        np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)

    def test_2d_larger(self):
        """Slightly larger brute-force check (15x15) for p=1.5."""
        rng = np.random.default_rng(999)
        labels = rng.integers(0, 3, size=(15, 15), dtype=np.uint8)
        got = edt_lp.edtp(labels, p=1.5, black_border=True, parallel=1)
        ref = _brute_force_lp(labels, 1.5, black_border=True)
        np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)

    def test_high_p_converges_to_linf(self):
        """As p→∞, Lp^p distance should be dominated by the max-axis term."""
        labels = np.zeros((9, 9), dtype=np.uint8)
        labels[2:7, 2:7] = 1
        # At center (4,4): distance to boundary is 3 on each axis
        # L∞ distance = 3, so Lp distance = max(dx, dy)
        # Lp^p = sum(|di|^p) → max(|di|)^p as p→∞
        # For (4,4): both axes contribute 3, so Lp^p ≈ 2 * 3^p
        # but the *nearest* boundary point has (dx=3,dy=0) so Lp^p = 3^p
        d_p10 = edt_lp.edt(labels, p=10.0, black_border=False)
        d_p50 = edt_lp.edt(labels, p=50.0, black_border=False)
        # Both should approach 3.0 at center as p→∞
        assert d_p10[4, 4] == pytest.approx(3.0, abs=0.1)
        assert d_p50[4, 4] == pytest.approx(3.0, abs=0.01)


# ---------------------------------------------------------------------------
# Known-value tests
# ---------------------------------------------------------------------------

class TestKnownValues:
    def test_p1_1d(self):
        labels = np.array([0, 1, 1, 1, 1, 0], dtype=np.uint8)
        got = edt_lp.edtp(labels, p=1.0, black_border=True)
        # L1^1: distance to nearest boundary
        expected = np.array([0, 1, 2, 2, 1, 0], dtype=np.float32)
        np.testing.assert_allclose(got, expected)

    def test_p2_1d(self):
        labels = np.array([0, 1, 1, 1, 1, 0], dtype=np.uint8)
        got = edt_lp.edtp(labels, p=2.0, black_border=True)
        # L2^2: squared distance
        expected = np.array([0, 1, 4, 4, 1, 0], dtype=np.float32)
        np.testing.assert_allclose(got, expected)

    def test_p3_1d(self):
        labels = np.array([0, 1, 1, 1, 0], dtype=np.uint8)
        got = edt_lp.edtp(labels, p=3.0, black_border=True)
        # L3^3: |d|^3 to nearest boundary
        expected = np.array([0, 1, 8, 1, 0], dtype=np.float32)
        np.testing.assert_allclose(got, expected)

    def test_p4_1d(self):
        labels = np.array([0, 1, 1, 1, 1, 1, 0], dtype=np.uint8)
        got = edt_lp.edtp(labels, p=4.0, black_border=True)
        # L4^4: |d|^4 to nearest boundary
        expected = np.array([0, 1, 16, 81, 16, 1, 0], dtype=np.float32)
        np.testing.assert_allclose(got, expected)

    def test_p1_1d_multilabel(self):
        labels = np.array([1, 1, 0, 2, 2, 2, 0], dtype=np.uint8)
        got = edt_lp.edtp(labels, p=1.0, black_border=True)
        # label 1: distances 1,1 (to border and to 0)
        # label 2: distances 1,2,1
        expected = np.array([1, 1, 0, 1, 2, 1, 0], dtype=np.float32)
        np.testing.assert_allclose(got, expected)

    def test_p1_2d_full_array(self):
        """L1: full 5x5 array with known distances."""
        labels = np.zeros((5, 5), dtype=np.uint8)
        labels[1:4, 1:4] = 1
        got = edt_lp.edtp(labels, p=1.0, black_border=False)
        # L1 distance = |dx| + |dy| to nearest bg voxel
        expected = np.zeros((5, 5), dtype=np.float32)
        expected[1, 1] = 1; expected[1, 2] = 1; expected[1, 3] = 1
        expected[2, 1] = 1; expected[2, 2] = 2; expected[2, 3] = 1
        expected[3, 1] = 1; expected[3, 2] = 1; expected[3, 3] = 1
        np.testing.assert_allclose(got, expected)

    def test_p2_2d_full_array(self):
        """L2^2: full 5x5 array with known squared distances."""
        labels = np.zeros((5, 5), dtype=np.uint8)
        labels[1:4, 1:4] = 1
        got = edt_lp.edtp(labels, p=2.0, black_border=False)
        expected = np.zeros((5, 5), dtype=np.float32)
        # Corners (1,1): nearest bg at (0,1) or (1,0) → dist²=1
        expected[1, 1] = 1; expected[1, 2] = 1; expected[1, 3] = 1
        expected[2, 1] = 1; expected[2, 3] = 1
        expected[3, 1] = 1; expected[3, 2] = 1; expected[3, 3] = 1
        # Center (2,2): nearest bg at (0,2) → 2²+0=4
        expected[2, 2] = 4
        np.testing.assert_allclose(got, expected)

    def test_p3_2d_full_array(self):
        """L3^3: full 5x5 array with known cubed distances."""
        labels = np.zeros((5, 5), dtype=np.uint8)
        labels[1:4, 1:4] = 1
        got = edt_lp.edtp(labels, p=3.0, black_border=False)
        expected = np.zeros((5, 5), dtype=np.float32)
        # Corner voxels (1,1): nearest bg at (0,1) or (1,0) → d=1 → 1^3=1
        expected[1, 1] = 1; expected[1, 2] = 1; expected[1, 3] = 1
        expected[2, 1] = 1; expected[2, 3] = 1
        expected[3, 1] = 1; expected[3, 2] = 1; expected[3, 3] = 1
        # Center (2,2): nearest bg at distance 2 on any axis
        # min over bg: (0,2) gives |2|^3+0=8, (2,0) gives 0+|2|^3=8
        # (1,0) gives 1+8=9, (0,1) gives 8+1=9, etc. → 8
        expected[2, 2] = 8
        np.testing.assert_allclose(got, expected)

    def test_p1_2d_diamond(self):
        """L1 distance shows diamond/taxicab pattern."""
        labels = np.zeros((7, 7), dtype=np.uint8)
        labels[1:6, 1:6] = 1
        got = edt_lp.edtp(labels, p=1.0, black_border=False)
        assert got[3, 3] == 3.0  # center: min(3, 3) = 3
        assert got[1, 1] == 1.0
        assert got[2, 2] == 2.0

    def test_p2_2d_euclidean(self):
        """L2^2 distance shows circular pattern."""
        labels = np.zeros((7, 7), dtype=np.uint8)
        labels[1:6, 1:6] = 1
        got = edt_lp.edtp(labels, p=2.0, black_border=False)
        assert got[2, 2] == pytest.approx(4.0)  # nearest boundary is 2 away on each axis
        assert got[3, 3] == pytest.approx(9.0)  # center: 3^2

    def test_anisotropy_p1(self):
        labels = np.array([0, 1, 1, 1, 0], dtype=np.uint8)
        got = edt_lp.edtp(labels, p=1.0, anisotropy=(3.0,), black_border=True)
        # w * |d| = 3*1=3, 3*2=6, 3*1=3
        expected = np.array([0, 3, 6, 3, 0], dtype=np.float32)
        np.testing.assert_allclose(got, expected)

    def test_anisotropy_p2(self):
        labels = np.array([0, 1, 1, 1, 0], dtype=np.uint8)
        got = edt_lp.edtp(labels, p=2.0, anisotropy=(3.0,), black_border=True)
        # w^2 * |d|^2 = 9*1=9, 9*4=36, 9*1=9
        expected = np.array([0, 9, 36, 9, 0], dtype=np.float32)
        np.testing.assert_allclose(got, expected)

    def test_anisotropy_p3(self):
        labels = np.array([0, 1, 1, 1, 0], dtype=np.uint8)
        got = edt_lp.edtp(labels, p=3.0, anisotropy=(2.0,), black_border=True)
        # w^p * |d|^p = 8*1=8, 8*8=64, 8*1=8
        expected = np.array([0, 8, 64, 8, 0], dtype=np.float32)
        np.testing.assert_allclose(got, expected)

    def test_p1_2d_anisotropy_full(self):
        """L1 with anisotropy (2.0, 1.0) on a small 2D array."""
        labels = np.zeros((5, 5), dtype=np.uint8)
        labels[1:4, 1:4] = 1
        anis = (2.0, 1.0)
        got = edt_lp.edtp(labels, p=1.0, anisotropy=anis, black_border=False)
        # (2,2): nearest bg at (2,0) → 2*0+1*2=2, or (0,2) → 2*2+1*0=4 → best=2
        assert got[2, 2] == pytest.approx(2.0)
        # (1,1): nearest bg at (1,0) → 2*0+1*1=1 or (0,1) → 2*1+1*0=2 → best=1
        assert got[1, 1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# edt() returns p-th root
# ---------------------------------------------------------------------------

class TestEdtRoot:
    def test_p2_sqrt(self):
        labels = np.array([0, 1, 1, 1, 0], dtype=np.uint8)
        dp = edt_lp.edtp(labels, p=2.0, black_border=True)
        d = edt_lp.edt(labels, p=2.0, black_border=True)
        np.testing.assert_allclose(d, np.sqrt(dp), rtol=1e-6)

    def test_p3_cbrt(self):
        labels = np.array([0, 1, 1, 1, 0], dtype=np.uint8)
        dp = edt_lp.edtp(labels, p=3.0, black_border=True)
        d = edt_lp.edt(labels, p=3.0, black_border=True)
        np.testing.assert_allclose(d, np.power(dp, 1.0 / 3.0), rtol=1e-6)

    def test_p1_identity(self):
        labels = np.array([0, 1, 1, 1, 0], dtype=np.uint8)
        dp = edt_lp.edtp(labels, p=1.0, black_border=True)
        d = edt_lp.edt(labels, p=1.0, black_border=True)
        np.testing.assert_array_equal(d, dp)  # p=1: no root needed


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_background(self):
        labels = np.zeros((10, 10), dtype=np.uint8)
        result = edt_lp.edtp(labels, p=1.5)
        np.testing.assert_array_equal(result, 0)

    def test_all_foreground_no_border(self):
        labels = np.ones((5, 5), dtype=np.uint8)
        result = edt_lp.edtp(labels, p=2.0, black_border=False)
        assert np.all(np.isinf(result))

    def test_all_foreground_with_border(self):
        labels = np.ones((5, 5), dtype=np.uint8)
        result = edt_lp.edtp(labels, p=2.0, black_border=True)
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)

    def test_single_voxel(self):
        labels = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8)
        for p in [1.0, 2.0, 3.0]:
            result = edt_lp.edtp(labels, p=p)
            assert result[1, 1] == 1.0  # distance 1 to boundary

    def test_p_validation(self):
        labels = np.array([0, 1, 0], dtype=np.uint8)
        with pytest.raises(ValueError, match="p must be >= 1"):
            edt_lp.edtp(labels, p=0.5)

    def test_0d_validation(self):
        with pytest.raises(ValueError):
            edt_lp.edtp(np.array(1, dtype=np.uint8), p=2.0)

    def test_empty_array(self):
        labels = np.zeros((0, 5), dtype=np.uint8)
        result = edt_lp.edtp(labels, p=2.0)
        assert result.shape == (0, 5)


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------

class TestAPI:
    def test_graph_api(self):
        labels = np.array([[0, 1, 1], [1, 1, 0]], dtype=np.uint8)
        graph = edt_lp.build_graph(labels)
        r1 = edt_lp.edtp(labels, p=1.5, black_border=True)
        r2 = edt_lp.edtp_graph(graph, p=1.5, black_border=True)
        np.testing.assert_array_equal(r1, r2)

    def test_edt_graph_api(self):
        labels = np.array([[0, 1, 1], [1, 1, 0]], dtype=np.uint8)
        graph = edt_lp.build_graph(labels)
        r1 = edt_lp.edt(labels, p=3.0, black_border=True)
        r2 = edt_lp.edt_graph(graph, p=3.0, black_border=True)
        np.testing.assert_allclose(r1, r2, rtol=1e-6)

    def test_sdf(self):
        labels = np.array([0, 0, 1, 1, 1, 0, 0], dtype=np.uint8)
        result = edt_lp.sdf(labels, p=2.0, black_border=True)
        assert result[0] < 0  # background: negative
        assert result[3] > 0  # foreground center: positive

    def test_threading_consistency(self):
        rng = np.random.default_rng(99)
        labels = rng.integers(0, 4, size=(100, 100), dtype=np.uint8)
        r1 = edt_lp.edtp(labels, p=1.5, parallel=1)
        r4 = edt_lp.edtp(labels, p=1.5, parallel=4)
        np.testing.assert_allclose(r1, r4, rtol=1e-6, atol=1e-6)

    def test_fortran_order(self):
        rng = np.random.default_rng(42)
        labels_c = rng.integers(0, 3, size=(30, 40), dtype=np.uint8)
        labels_f = np.asfortranarray(labels_c)
        r_c = edt_lp.edtp(labels_c, p=1.5, parallel=1)
        r_f = edt_lp.edtp(labels_f, p=1.5, parallel=1)
        np.testing.assert_allclose(r_c, r_f, rtol=1e-5, atol=1e-5)

    def test_edtsq_alias(self):
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 3, size=(50, 50), dtype=np.uint8)
        ref = edt.edtsq(labels, parallel=1)
        got = edt_lp.edtsq(labels, parallel=1)
        np.testing.assert_array_equal(ref, got)

    def test_edtsq_graph_alias(self):
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 3, size=(30, 30), dtype=np.uint8)
        graph = edt_lp.build_graph(labels)
        ref = edt_lp.edtp_graph(graph, p=2.0, parallel=1)
        got = edt_lp.edtsq_graph(graph, parallel=1)
        np.testing.assert_array_equal(ref, got)

    def test_sdfsq(self):
        labels = np.array([0, 0, 1, 1, 1, 0, 0], dtype=np.uint8)
        result = edt_lp.sdfsq(labels)
        assert result[0] < 0
        assert result[3] > 0


# ---------------------------------------------------------------------------
# expand_labels tests
# ---------------------------------------------------------------------------

class TestExpandLabels:
    def test_p2_parity_with_src(self):
        """expand_labels(p=2) must match edt.expand_labels exactly."""
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 5, size=(50, 50), dtype=np.uint32)
        ref = edt.expand_labels(labels, parallel=1)
        got = edt_lp.expand_labels(labels, p=2.0, parallel=1)
        np.testing.assert_array_equal(ref, got)

    def test_p2_parity_3d(self):
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 3, size=(20, 20, 20), dtype=np.uint32)
        ref = edt.expand_labels(labels, parallel=1)
        got = edt_lp.expand_labels(labels, p=2.0, parallel=1)
        np.testing.assert_array_equal(ref, got)

    def test_p2_parity_with_border(self):
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 5, size=(30, 30), dtype=np.uint32)
        ref = edt.expand_labels(labels, black_border=True, parallel=1)
        got = edt_lp.expand_labels(labels, p=2.0, black_border=True, parallel=1)
        np.testing.assert_array_equal(ref, got)

    def test_p2_parity_anisotropy(self):
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 4, size=(30, 40), dtype=np.uint32)
        anis = (2.0, 0.5)
        ref = edt.expand_labels(labels, anisotropy=anis, parallel=1)
        got = edt_lp.expand_labels(labels, p=2.0, anisotropy=anis, parallel=1)
        np.testing.assert_array_equal(ref, got)

    def test_1d(self):
        labels = np.array([0, 0, 1, 0, 0, 2, 0, 0], dtype=np.uint32)
        got = edt_lp.expand_labels(labels, p=2.0, parallel=1)
        # Each zero should be assigned to nearest seed
        assert got[0] == 1  # closer to seed at idx 2
        assert got[3] == 1  # equidistant: 1 from idx 2, 2 from idx 5 → 1
        assert got[7] == 2  # closer to seed at idx 5

    def test_all_zero(self):
        labels = np.zeros((10, 10), dtype=np.uint32)
        got = edt_lp.expand_labels(labels, p=2.0, parallel=1)
        np.testing.assert_array_equal(got, 0)

    def test_return_features(self):
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 5, size=(30, 30), dtype=np.uint32)
        ref_labels, ref_feats = edt.expand_labels(labels, parallel=1, return_features=True)
        got_labels, got_feats = edt_lp.expand_labels(labels, p=2.0, parallel=1, return_features=True)
        np.testing.assert_array_equal(ref_labels, got_labels)
        np.testing.assert_array_equal(ref_feats, got_feats)

    @pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
    def test_lp_runs(self, p):
        """Verify expand_labels produces valid output for various p."""
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 4, size=(20, 20), dtype=np.uint32)
        got = edt_lp.expand_labels(labels, p=p, parallel=1)
        # Must have same shape
        assert got.shape == labels.shape
        # Seeds must keep their labels
        seed_mask = labels != 0
        np.testing.assert_array_equal(got[seed_mask], labels[seed_mask])
        # Non-seeds must get a non-zero label (all are reachable)
        assert np.all(got > 0)

    def test_lp_diamond_vs_circle(self):
        """L1 expand should produce diamond Voronoi regions, L2 circular."""
        # Seeds must NOT be axis-aligned for boundary shape to differ
        labels = np.zeros((31, 31), dtype=np.uint32)
        labels[5, 5] = 1    # top-left seed
        labels[25, 25] = 2  # bottom-right seed (diagonal)
        # L1: boundary is a diagonal line (taxicab equidistant)
        # L2: boundary is perpendicular bisector (Euclidean equidistant)
        # These differ for diagonally placed seeds.
        exp_l1 = edt_lp.expand_labels(labels, p=1.0, parallel=1)
        exp_l2 = edt_lp.expand_labels(labels, p=2.0, parallel=1)
        # They should differ (different Voronoi boundaries)
        assert not np.array_equal(exp_l1, exp_l2)


# ---------------------------------------------------------------------------
# feature_transform tests
# ---------------------------------------------------------------------------

class TestFeatureTransform:
    def test_p2_parity_with_src(self):
        """feature_transform(p=2) must match edt.feature_transform exactly."""
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 5, size=(30, 30), dtype=np.uint32)
        ref = edt.feature_transform(labels, parallel=1)
        got = edt_lp.feature_transform(labels, p=2.0, parallel=1)
        np.testing.assert_array_equal(ref, got)

    def test_p2_parity_with_distances(self):
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 5, size=(30, 30), dtype=np.uint32)
        ref_feat, ref_dist = edt.feature_transform(labels, parallel=1, return_distances=True)
        got_feat, got_dist = edt_lp.feature_transform(labels, p=2.0, parallel=1, return_distances=True)
        np.testing.assert_array_equal(ref_feat, got_feat)
        np.testing.assert_allclose(ref_dist, got_dist, rtol=1e-6)

    @pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
    def test_lp_runs(self, p):
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 4, size=(20, 20), dtype=np.uint32)
        feat = edt_lp.feature_transform(labels, p=p, parallel=1)
        assert feat.shape == labels.shape
        # Each seed should point to itself
        flat_labels = labels.ravel()
        flat_feat = feat.ravel()
        for idx in range(flat_labels.size):
            if flat_labels[idx] != 0:
                assert flat_feat[idx] == idx

    def test_empty(self):
        labels = np.zeros((0, 5), dtype=np.uint32)
        feat = edt_lp.feature_transform(labels, p=2.0, parallel=1)
        assert feat.shape == (0, 5)

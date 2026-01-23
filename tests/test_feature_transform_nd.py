import numpy as np

import edt


def _bruteforce_nearest(shape, seeds, anisotropy):
    """Return linear index of nearest seed per voxel (C-order).

    Tie-breaks by choosing the larger seed index to match expand_labels 1D.
    """
    anis = np.asarray(anisotropy, dtype=np.float64)
    coords = np.indices(shape, dtype=np.float64).reshape(len(shape), -1).T
    seed_coords = np.array([s[0] for s in seeds], dtype=np.float64)
    seed_lin = np.array(
        [np.ravel_multi_index(tuple(s[0]), shape, order="C") for s in seeds],
        dtype=np.int64,
    )
    # Compute squared distances with anisotropy scaling.
    diffs = coords[:, None, :] - seed_coords[None, :, :]
    diffs *= anis[None, None, :]
    d2 = np.sum(diffs * diffs, axis=2)
    # Tie-break toward the larger seed index.
    nearest = np.empty((d2.shape[0],), dtype=np.int64)
    for i in range(d2.shape[0]):
        row = d2[i]
        m = np.min(row)
        candidates = np.flatnonzero(row == m)
        if candidates.size == 1:
            nearest[i] = candidates[0]
        else:
            # Choose the largest linear index among tied seeds.
            best = candidates[np.argmax(seed_lin[candidates])]
            nearest[i] = best
    return seed_lin[nearest].reshape(shape)


def test_feature_transform_matches_bruteforce_2d():
    shape = (7, 9)
    arr = np.zeros(shape, dtype=np.uint32)
    seeds = [((1, 1), 10), ((5, 6), 20)]
    for coord, label in seeds:
        arr[coord] = label
    anis = (1.0, 2.0)

    feats = edt.feature_transform(arr, anisotropy=anis, parallel=1)
    expected = _bruteforce_nearest(shape, seeds, anis)
    np.testing.assert_array_equal(feats, expected)
    np.testing.assert_array_equal(arr.ravel()[feats], arr.ravel()[expected])


def test_feature_transform_return_distances_matches_edtsq_nd():
    shape = (6, 6, 4)
    arr = np.zeros(shape, dtype=np.uint32)
    arr[1, 1, 1] = 5
    arr[4, 2, 3] = 7
    anis = (1.0, 1.5, 2.0)

    feats, dist = edt.feature_transform(
        arr, anisotropy=anis, parallel=1, return_distances=True
    )
    ref = edt.edtsq_nd((arr != 0).astype(np.uint8), anisotropy=anis, parallel=1)
    np.testing.assert_allclose(dist, ref, rtol=1e-6, atol=1e-6)
    assert feats.shape == arr.shape


def test_expand_labels_return_features_consistent_with_feature_transform():
    shape = (5, 8)
    arr = np.zeros(shape, dtype=np.uint32)
    arr[0, 0] = 1
    arr[4, 7] = 2
    anis = (1.0, 1.0)

    labels, feats = edt.expand_labels(
        arr, anisotropy=anis, parallel=1, return_features=True
    )
    ft = edt.feature_transform(arr, anisotropy=anis, parallel=1)
    np.testing.assert_array_equal(feats, ft)
    np.testing.assert_array_equal(labels, arr.ravel()[feats].reshape(shape))


def test_feature_transform_features_dtype_options():
    shape = (4, 4)
    arr = np.zeros(shape, dtype=np.uint32)
    arr[0, 0] = 1
    arr[3, 3] = 2

    feats_u32 = edt.feature_transform(arr, features_dtype="uint32", parallel=1)
    feats_up = edt.feature_transform(arr, features_dtype="uintp", parallel=1)
    assert feats_u32.dtype == np.uint32
    assert feats_up.dtype == np.uintp


def test_feature_transform_anisotropy_length_mismatch_raises():
    arr = np.zeros((4, 4), dtype=np.uint32)
    with np.testing.assert_raises(ValueError):
        edt.feature_transform(arr, anisotropy=(1.0, 2.0, 3.0))


def test_feature_transform_1d_matches_bruteforce():
    arr = np.zeros((8,), dtype=np.uint32)
    arr[2] = 3
    arr[6] = 5
    anis = (1.0,)

    feats = edt.feature_transform(arr, anisotropy=anis, parallel=1)
    seeds = [((2,), 3), ((6,), 5)]
    expected = _bruteforce_nearest(arr.shape, seeds, anis)
    np.testing.assert_array_equal(feats, expected)

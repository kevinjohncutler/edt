import numpy as np

def make_label_matrix(N: int, M: int) -> np.ndarray:
    """
    General ND label matrix.
    
    Shape = (2*M,)*N
    Each axis is split into two halves of length M.
    The label is the binary code of the half-indices.
    
    Example:
      N=1 → [0...0,1...1]
      N=2 → quadrants labeled 0..3
      N=3 → octants labeled 0..7
      N=4 → 16 hyper-quadrants labeled 0..15
    """
    if N < 1:
        raise ValueError("N must be >=1")
    # build index grids
    grids = np.ogrid[tuple(slice(0,2*M) for _ in range(N))]
    labels = np.zeros((2*M,)*N, dtype=int)
    for axis, g in enumerate(grids):
        half = (g // M).astype(int)     # 0 or 1
        labels += half << axis          # bit-shift
    return labels

def test_edt_consistency():
    """Test that edt functions give consistent results across dimensions"""
    
    print("="*60)
    print("EDT CONSISTENCY TEST")
    print("="*60)
    
    import edt
    legacy = getattr(edt, "legacy", None)
    has_legacy = bool(legacy) and getattr(legacy, "available", lambda: False)()
    
    for ndim in [1, 2, 3]:
        print(f"\n--- Testing {ndim}D ---")
        
        # Test with M=3 (smaller for readability)
        M = 3
        masks = make_label_matrix(ndim, M)
        
        print(f"Input shape: {masks.shape}")
        print(f"Input size: {masks.size}")
        print(f"Unique labels: {np.unique(masks)}")
        
        if has_legacy:
            if ndim == 1:
                dt_orig = legacy.edt1d(masks)
            elif ndim == 2:
                dt_orig = legacy.edt2d(masks)
            else:
                dt_orig = legacy.edt3d(masks)

            print(f"Original edt{ndim}d: range {dt_orig.min():.3f} to {dt_orig.max():.3f}")
            print(f"Expected max: {M} (side length)")
            print(f"Max matches expected: {abs(dt_orig.max() - M) < 1e-6}")
        else:
            dt_orig = None
            print("Legacy edt.legacy module unavailable; skipping specialized comparison.")

        dt_nd = edt.edt_nd(masks)
        print(f"ND edt_nd: range {dt_nd.min():.3f} to {dt_nd.max():.3f}")
        print(f"Max matches expected: {abs(dt_nd.max() - M) < 1e-6}")

        if dt_orig is not None:
            diff = np.abs(dt_orig - dt_nd)
            max_diff = diff.max()
            print(f"Max difference: {max_diff:.6f}")

            if max_diff < 1e-6:
                print("Results match within tolerance.")
            else:
                print("Results differ beyond tolerance.")
                max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
                print(f"Max difference at position {max_diff_idx}:")
                print(f"  Original: {dt_orig[max_diff_idx]:.3f}")
                print(f"  ND: {dt_nd[max_diff_idx]:.3f}")
                print(f"  Input value: {masks[max_diff_idx]}")

if __name__ == "__main__":
    test_edt_consistency()

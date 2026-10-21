"""NumPy-based centroid voxel downsampling."""

import numpy as np


def _voxel_keys(voxel_indices):
    """Encode (N, 3) int64 voxel indices as collision-free int64 keys.

    Keys are x-major, so sorting them gives the same order as
    np.unique(voxel_indices, axis=0), which is ~5x slower.

    Returns None if the occupied index range does not fit in int64.
    """
    mins = voxel_indices.min(axis=0)
    maxs = voxel_indices.max(axis=0)

    # Python ints: exact, no overflow while checking.
    extents = [int(hi) - int(lo) + 1 for lo, hi in zip(mins, maxs)]
    if extents[0] * extents[1] * extents[2] > np.iinfo(np.int64).max:
        return None

    offsets = voxel_indices - mins
    return (
        (offsets[:, 0] * extents[1] + offsets[:, 1]) * extents[2]
        + offsets[:, 2]
    )


def voxel_downsample(points, voxel_size=0.2):
    """Downsample XYZ points using one centroid per occupied voxel.

    Args:
        points: (N, 3) array of XYZ coordinates in metres.
        voxel_size: Positive voxel edge length in metres.

    Returns:
        (M, 3) float32 array, where M <= N.
    """
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be finite and > 0")

    pts = np.asarray(points)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(
            f"Expected points shape (N, 3), got {pts.shape}"
        )

    if pts.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    # Remove non-finite points before computing voxel coordinates.
    pts = pts[np.isfinite(pts).all(axis=1)]

    if len(pts) == 0:
        return np.empty((0, 3), dtype=np.float32)

    pts = np.asarray(pts, dtype=np.float64)

    # Floor is necessary for correct negative-coordinate binning.
    scaled = np.floor(pts / voxel_size)
    if not np.isfinite(scaled).all():
        raise ValueError("Voxel indices must be finite")

    limit = np.iinfo(np.int64).max
    if np.any(np.abs(scaled) >= limit):
        raise ValueError("Voxel indices exceed int64 range")

    voxel_indices = scaled.astype(np.int64)

    # Assign each point to a unique 3D voxel.
    keys = _voxel_keys(voxel_indices)

    if keys is not None:
        _, inverse, counts = np.unique(
            keys,
            return_inverse=True,
            return_counts=True,
        )
    else:
        # Key range exceeds int64; fall back to row-wise grouping.
        _, inverse, counts = np.unique(
            voxel_indices,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )

    # Compute the centroid of each occupied voxel.
    sums = np.column_stack([
        np.bincount(
            inverse,
            weights=pts[:, axis],
            minlength=len(counts),
        )
        for axis in range(3)
    ])

    centroids = sums / counts[:, None]

    return np.ascontiguousarray(
        centroids,
        dtype=np.float32,
    )

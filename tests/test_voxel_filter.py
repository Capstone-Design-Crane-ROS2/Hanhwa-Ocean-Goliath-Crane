"""Unit tests for centroid voxel downsampling."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "lidar"),
)

from voxel_filter import _voxel_keys, voxel_downsample


def reference_downsample(points, voxel_size):
    """Original row-wise np.unique(axis=0) implementation."""
    pts = np.asarray(points, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    indices = np.floor(pts / voxel_size).astype(np.int64)

    _, inverse, counts = np.unique(
        indices, axis=0, return_inverse=True, return_counts=True,
    )
    sums = np.column_stack([
        np.bincount(inverse, weights=pts[:, axis],
                    minlength=len(counts))
        for axis in range(3)
    ])
    return (sums / counts[:, None]).astype(np.float32)


def by_voxel(centroids, voxel_size):
    """Map voxel index -> centroid, for order-independent comparison."""
    keys = np.floor(
        centroids.astype(np.float64) / voxel_size
    ).astype(np.int64)
    return {tuple(key): c for key, c in zip(keys, centroids)}


class TestVoxelFilter(unittest.TestCase):

    def test_same_voxel_centroid(self):
        points = np.array([
            [0.01, 0.02, 0.03],
            [0.03, 0.04, 0.05],
        ], dtype=np.float32)

        result = voxel_downsample(points, 0.2)

        self.assertEqual(result.shape, (1, 3))
        np.testing.assert_allclose(
            result[0],
            [0.02, 0.03, 0.04],
            atol=1e-6,
        )

    def test_different_voxels(self):
        points = np.array([
            [0.01, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.45, 0.0, 0.0],
        ], dtype=np.float32)

        result = voxel_downsample(points, 0.2)
        self.assertEqual(len(result), 3)

    def test_negative_coordinates(self):
        points = np.array([
            [-0.01, 0.0, 0.0],
            [0.01, 0.0, 0.0],
        ], dtype=np.float32)

        result = voxel_downsample(points, 0.2)
        self.assertEqual(len(result), 2)

    def test_remove_nan_and_inf(self):
        points = np.array([
            [0.01, 0.0, 0.0],
            [np.nan, 0.0, 0.0],
            [np.inf, 0.0, 0.0],
        ], dtype=np.float32)

        result = voxel_downsample(points, 0.2)

        self.assertEqual(len(result), 1)
        self.assertTrue(np.isfinite(result).all())

    def test_empty_input(self):
        points = np.empty((0, 3), dtype=np.float32)
        result = voxel_downsample(points, 0.2)

        self.assertEqual(result.shape, (0, 3))
        self.assertEqual(result.dtype, np.float32)

    def test_invalid_voxel_size(self):
        points = np.array([[0, 0, 0]], dtype=np.float32)

        for size in (0, -0.2, np.nan, np.inf):
            with self.assertRaises(ValueError):
                voxel_downsample(points, size)

    def test_invalid_shape(self):
        points = np.zeros((3, 2), dtype=np.float32)

        with self.assertRaises(ValueError):
            voxel_downsample(points, 0.2)

    def test_output_dtype(self):
        points = np.array([
            [1.0, 2.0, 3.0],
        ], dtype=np.float64)

        result = voxel_downsample(points, 0.2)
        self.assertEqual(result.dtype, np.float32)

    def test_voxel_boundary(self):
        # A point exactly on a boundary belongs to the upper voxel.
        points = np.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ], dtype=np.float64)

        result = voxel_downsample(points, 0.2)

        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(
            result, [[0.05, 0.0, 0.0], [0.2, 0.0, 0.0]], atol=1e-6,
        )

    def test_negative_boundary(self):
        # floor(-0.2 / 0.2) = -1, same voxel as -0.1;
        # -0.21 falls in voxel -2.
        points = np.array([
            [-0.2, 0.0, 0.0],
            [-0.1, 0.0, 0.0],
            [-0.21, 0.0, 0.0],
        ], dtype=np.float64)

        result = voxel_downsample(points, 0.2)

        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(
            result, [[-0.21, 0.0, 0.0], [-0.15, 0.0, 0.0]], atol=1e-6,
        )

    def test_matches_reference_on_random_clouds(self):
        rng = np.random.default_rng(0)

        for voxel_size in (0.05, 0.2, 0.5, 1.0):
            for _ in range(5):
                points = rng.uniform(
                    -60.0, 60.0, size=(4000, 3)
                ).astype(np.float32)
                # Dense duplicates so many voxels hold several points.
                points[:1000] = points[1000:2000] + rng.normal(
                    0.0, 0.01, size=(1000, 3)
                ).astype(np.float32)
                points[::997] = np.nan

                result = voxel_downsample(points, voxel_size)
                expected = reference_downsample(points, voxel_size)

                self.assertEqual(len(result), len(expected))
                got = by_voxel(result, voxel_size)
                want = by_voxel(expected, voxel_size)
                self.assertEqual(got.keys(), want.keys())
                for key, centroid in want.items():
                    np.testing.assert_allclose(
                        got[key], centroid, rtol=0, atol=1e-5,
                    )

    def test_large_coordinates(self):
        points = np.array([
            [1.0e6, -1.0e6, 1.0e3],
            [1.0e6 + 0.05, -1.0e6 + 0.05, 1.0e3 + 0.05],
            [-1.0e6, 1.0e6, -1.0e3],
        ], dtype=np.float64)

        result = voxel_downsample(points, 0.2)

        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(
            result, reference_downsample(points, 0.2)
        )

    def test_key_overflow_falls_back(self):
        # Index range ~1e18 per axis cannot be packed into int64.
        points = np.array([
            [-1.0e18, -1.0e18, -1.0e18],
            [1.0e18, 1.0e18, 1.0e18],
            [1.0e18, 1.0e18, 1.0e18],
        ], dtype=np.float64)
        indices = np.floor(points).astype(np.int64)

        self.assertIsNone(_voxel_keys(indices))

        result = voxel_downsample(points, 1.0)

        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(
            result, reference_downsample(points, 1.0)
        )

    def test_keys_are_collision_free(self):
        rng = np.random.default_rng(1)
        indices = rng.integers(-50, 50, size=(50000, 3))

        keys = _voxel_keys(indices)

        self.assertEqual(
            len(np.unique(keys)),
            len(np.unique(indices, axis=0)),
        )


if __name__ == "__main__":
    unittest.main()

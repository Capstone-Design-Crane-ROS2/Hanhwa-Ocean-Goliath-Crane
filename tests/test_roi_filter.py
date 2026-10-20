"""roi_filter 단위 테스트 (ROS2 불필요).

실행: python3 -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lidar'))

from roi_filter import DEFAULT_ROI, RoiBounds, filter_roi  # noqa: E402

BOUNDS = RoiBounds(x_min=-50, x_max=50, y_min=-30, y_max=30, z_min=-5, z_max=15)


class TestRoiFilter(unittest.TestCase):
    def test_default_roi_values(self):
        self.assertEqual(DEFAULT_ROI, BOUNDS)

    def test_inside_points_kept(self):
        pts = np.array([[0, 0, 0], [10, -5, 3], [-49.9, 29.9, 14.9]], dtype=np.float32)
        out, stats = filter_roi(pts, BOUNDS)
        np.testing.assert_array_equal(out, pts)
        self.assertEqual(stats.input_count, 3)
        self.assertEqual(stats.output_count, 3)
        self.assertEqual(stats.outside_count, 0)

    def test_outside_points_removed(self):
        pts = np.array([
            [1, 1, 1],        # 내부
            [50.1, 0, 0],     # x 초과
            [-50.1, 0, 0],    # x 미만
            [0, 30.1, 0],     # y 초과
            [0, -30.1, 0],    # y 미만
            [0, 0, 15.1],     # z 초과
            [0, 0, -5.1],     # z 미만
        ], dtype=np.float64)
        out, stats = filter_roi(pts, BOUNDS)
        np.testing.assert_array_equal(out, [[1, 1, 1]])
        self.assertEqual(stats.outside_count, 6)
        self.assertEqual(stats.output_count, 1)

    def test_boundary_points_included(self):
        pts = np.array([
            [50, 30, 15],
            [-50, -30, -5],
            [50, 0, 0],
            [0, -30, 15],
        ], dtype=np.float32)
        out, stats = filter_roi(pts, BOUNDS)
        self.assertEqual(stats.output_count, 4)
        np.testing.assert_array_equal(out, pts)

    def test_nan_inf_removed(self):
        pts = np.array([
            [1, 2, 3],
            [np.nan, 0, 0],
            [0, np.inf, 0],
            [0, 0, -np.inf],
            [np.nan, np.nan, np.nan],
        ], dtype=np.float32)
        out, stats = filter_roi(pts, BOUNDS)
        np.testing.assert_array_equal(out, [[1, 2, 3]])
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(stats.invalid_count, 4)
        self.assertEqual(stats.outside_count, 0)
        self.assertEqual(stats.output_count, 1)

    def test_empty_input(self):
        for empty in (np.empty((0, 3), dtype=np.float32), np.array([]), []):
            out, stats = filter_roi(empty, BOUNDS)
            self.assertEqual(out.shape, (0, 3))
            self.assertEqual(stats.input_count, 0)
            self.assertEqual(stats.output_count, 0)

    def test_all_points_removed_shape(self):
        out, _ = filter_roi(np.array([[100, 100, 100]], dtype=np.float32), BOUNDS)
        self.assertEqual(out.shape, (0, 3))

    def test_shape_and_dtype(self):
        rng = np.random.default_rng(0)
        pts = rng.uniform(-100, 100, size=(1000, 3)).astype(np.float32)
        out, stats = filter_roi(pts, BOUNDS)
        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape[1], 3)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.shape[0], stats.output_count)
        self.assertEqual(
            stats.input_count,
            stats.invalid_count + stats.outside_count + stats.output_count,
        )
        lo, hi = BOUNDS.as_arrays()
        self.assertTrue(((out >= lo) & (out <= hi)).all())

    def test_input_not_modified(self):
        pts = np.array([[1, 1, 1], [np.nan, 0, 0], [99, 0, 0]], dtype=np.float32)
        before = pts.copy()
        filter_roi(pts, BOUNDS)
        np.testing.assert_array_equal(pts, before)

    def test_invalid_shape_raises(self):
        with self.assertRaises(ValueError):
            filter_roi(np.zeros((5, 2)), BOUNDS)
        with self.assertRaises(ValueError):
            filter_roi(np.zeros(3), BOUNDS)

    def test_invalid_bounds_raises(self):
        with self.assertRaises(ValueError):
            RoiBounds(x_min=10, x_max=-10, y_min=-1, y_max=1, z_min=-1, z_max=1)
        with self.assertRaises(ValueError):
            RoiBounds(x_min=-np.inf, x_max=1, y_min=-1, y_max=1, z_min=-1, z_max=1)

    def test_custom_bounds(self):
        small = RoiBounds(x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1)
        pts = np.array([[0.5, 0.5, 0.5], [2, 0.5, 0.5]], dtype=np.float32)
        out, _ = filter_roi(pts, small)
        np.testing.assert_array_equal(out, [[0.5, 0.5, 0.5]])


if __name__ == '__main__':
    unittest.main()

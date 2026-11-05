"""Unit tests for Euclidean object candidate extraction."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "lidar"),
)

from cluster_extractor import (
    ClusterParams,
    EuclideanClusterExtractor,
    label_components,
)


def blob(center, count=30, spread=0.15, seed=0):
    rng = np.random.default_rng(seed)
    return np.asarray(center) + rng.uniform(-spread, spread, (count, 3))


def extractor(tolerance=0.6, min_points=5, max_points=3000):
    return EuclideanClusterExtractor(
        ClusterParams(tolerance, min_points, max_points)
    )


class ExtractTest(unittest.TestCase):
    def test_empty_cloud(self):
        self.assertEqual(extractor().extract(np.empty((0, 3))), [])

    def test_empty_list(self):
        self.assertEqual(extractor().extract([]), [])

    def test_all_non_finite(self):
        points = np.full((6, 3), np.nan)
        self.assertEqual(extractor().extract(points), [])

    def test_single_object(self):
        groups = extractor().extract(blob([5, 0, 0], 40))
        self.assertEqual(len(groups), 1)
        np.testing.assert_array_equal(groups[0], np.arange(40))

    def test_multiple_objects_are_separated(self):
        a = blob([5, 0, 0], 30, seed=1)
        b = blob([5, 3, 0], 20, seed=2)
        c = blob([-8, 1, 1], 25, seed=3)
        groups = extractor().extract(np.vstack([a, b, c]))
        self.assertEqual(
            [set(g.tolist()) for g in groups],
            [set(range(30)), set(range(30, 50)), set(range(50, 75))],
        )

    def test_gap_at_tolerance_boundary(self):
        points = np.array([[0.0, 0, 0], [0.5, 0, 0]])
        self.assertEqual(
            len(extractor(tolerance=0.6, min_points=1).extract(points)), 1
        )
        self.assertEqual(
            len(extractor(tolerance=0.4, min_points=1).extract(points)), 2
        )

    def test_chain_links_transitively(self):
        # Each neighbour within tolerance, the ends 4.5 m apart: one object.
        points = np.column_stack([
            np.arange(10) * 0.5, np.zeros(10), np.zeros(10),
        ])
        self.assertEqual(len(extractor().extract(points)), 1)

    def test_bridge_merges_two_objects(self):
        # Documents the known limit: a touching structure merges objects.
        truck = blob([0, 0, 0], 30, seed=1)
        wall = blob([2, 0, 0], 30, seed=2)
        bridge = np.column_stack([
            np.linspace(0, 2, 9), np.zeros(9), np.zeros(9),
        ])
        self.assertEqual(
            len(extractor().extract(np.vstack([truck, wall]))), 2
        )
        self.assertEqual(
            len(extractor().extract(np.vstack([truck, wall, bridge]))), 1
        )

    def test_min_points_boundary(self):
        points = blob([0, 0, 0], 5)
        self.assertEqual(len(extractor(min_points=5).extract(points)), 1)
        self.assertEqual(extractor(min_points=6).extract(points), [])

    def test_max_points_boundary(self):
        points = blob([0, 0, 0], 50)
        self.assertEqual(len(extractor(max_points=50).extract(points)), 1)
        self.assertEqual(extractor(max_points=49).extract(points), [])

    def test_size_limits_keep_other_clusters(self):
        small = blob([0, 0, 0], 3, seed=1)
        good = blob([10, 0, 0], 20, seed=2)
        big = blob([20, 0, 0], 100, seed=3)
        groups = extractor(min_points=5, max_points=50).extract(
            np.vstack([small, good, big])
        )
        self.assertEqual(len(groups), 1)
        np.testing.assert_array_equal(groups[0], np.arange(3, 23))

    def test_non_finite_points_keep_original_indices(self):
        points = blob([1, 1, 1], 10)
        points = np.insert(points, 3, [np.nan, 0, 0], axis=0)
        points = np.insert(points, 0, [np.inf, 0, 0], axis=0)
        groups = extractor().extract(points)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 10)
        self.assertTrue(np.isfinite(points[groups[0]]).all())
        self.assertNotIn(0, groups[0])
        self.assertNotIn(4, groups[0])

    def test_groups_are_disjoint_int_indices(self):
        points = np.vstack([blob([i * 5, 0, 0], 12, seed=i) for i in range(5)])
        groups = extractor().extract(points)
        joined = np.concatenate(groups)
        self.assertEqual(len(joined), len(set(joined.tolist())))
        self.assertTrue(all(g.dtype == np.int64 for g in groups))

    def test_deterministic(self):
        points = np.vstack([blob([i * 5, 0, 0], 12, seed=i) for i in range(5)])
        first = extractor().extract(points)
        second = extractor().extract(points)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_float32_input(self):
        points = blob([0, 0, 0], 10).astype(np.float32)
        self.assertEqual(len(extractor().extract(points)), 1)

    def test_rejects_bad_shape(self):
        with self.assertRaises(ValueError):
            extractor().extract(np.zeros((5, 2)))

    def test_ignores_context(self):
        points = blob([0, 0, 0], 10)
        self.assertEqual(
            len(extractor().extract(points, context={"boxes": []})), 1
        )


class LabelComponentsTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(len(label_components(np.empty((0, 3)), 0.5)), 0)

    def test_labels(self):
        labels = label_components(
            np.array([[0.0, 0, 0], [0.1, 0, 0], [5.0, 0, 0]]), 0.5
        )
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])


class ParamsTest(unittest.TestCase):
    def test_invalid(self):
        for kwargs in (
            {"tolerance": 0.0},
            {"tolerance": -1.0},
            {"tolerance": float("nan")},
            {"min_points": 0},
            {"min_points": 10, "max_points": 5},
        ):
            with self.assertRaises(ValueError):
                ClusterParams(**kwargs)


if __name__ == "__main__":
    unittest.main()

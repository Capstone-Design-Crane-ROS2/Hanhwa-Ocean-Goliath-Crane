"""Unit tests for object position, distance, size and box estimation."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "lidar"),
)

from object_estimator import (
    EstimatorParams,
    ReferencePose,
    box_corners,
    estimate_object,
    estimate_objects,
    height_band_mask,
)


# Dataset sensor pose: world <- OS1_A, z axis tilted 20 deg from vertical.
DATASET_POSE = ReferencePose(
    (0.17364817766693036, 0.0, 0.9848077530122081, 0.0),
    (396.8843504349122, 0.052, 5.669106866204646),
)


def grid_box(lo, hi, steps=5):
    """Regular grid of points filling an axis-aligned box."""
    axes = [np.linspace(a, b, steps) for a, b in zip(lo, hi)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


class SingleObjectTest(unittest.TestCase):
    def test_center_size_and_distance(self):
        points = grid_box([3, -1, 0], [5, 1, 2])
        obj = estimate_object(points, np.arange(len(points)), 7)
        self.assertEqual(obj.id, 7)
        self.assertEqual(obj.point_count, len(points))
        np.testing.assert_allclose(obj.bbox_center, [4, 0, 1])
        np.testing.assert_allclose(obj.center, [4, 0, 1])
        np.testing.assert_allclose(obj.size, [2, 2, 2])
        self.assertAlmostEqual(obj.distance, np.sqrt(17))
        self.assertAlmostEqual(obj.horizontal_distance, 4.0)

    def test_nearest_distance(self):
        points = np.array([[3.0, 4.0, 0.0], [6.0, 8.0, 0.0]])
        obj = estimate_object(points, [0, 1], 0)
        self.assertAlmostEqual(obj.nearest_distance, 5.0)
        self.assertAlmostEqual(obj.distance, 7.5)

    def test_bbox_center_vs_centroid_with_uneven_density(self):
        # Most points on the near face: the mean is pulled to it, the box
        # center is not.
        near = np.column_stack([
            np.full(90, 10.0),
            np.linspace(-1, 1, 90),
            np.zeros(90),
        ])
        far = np.array([[14.0, 0.0, 0.0]] * 10)
        points = np.vstack([near, far])
        indices = np.arange(100)

        by_box = estimate_object(points, indices, 0, center_method="bbox")
        by_mean = estimate_object(points, indices, 0, center_method="centroid")

        np.testing.assert_allclose(by_box.center, [12, 0, 0], atol=1e-9)
        np.testing.assert_allclose(by_mean.center, [10.4, 0, 0], atol=1e-9)
        # Both values are always available.
        np.testing.assert_allclose(by_box.centroid, by_mean.centroid)
        np.testing.assert_allclose(by_mean.bbox_center, by_box.bbox_center)
        self.assertAlmostEqual(by_box.distance, 12.0)
        self.assertAlmostEqual(by_mean.distance, 10.4)

    def test_only_selected_points_are_used(self):
        points = np.array([[1.0, 0, 0], [2.0, 0, 0], [100.0, 0, 0]])
        obj = estimate_object(points, np.array([0, 1]), 0)
        np.testing.assert_allclose(obj.bbox_max, [2, 0, 0])
        np.testing.assert_array_equal(obj.point_indices, [0, 1])

    def test_single_point_has_zero_size(self):
        obj = estimate_object(np.array([[1.0, 2.0, 2.0]]), [0], 0)
        np.testing.assert_allclose(obj.size, 0.0)
        self.assertAlmostEqual(obj.distance, 3.0)

    def test_empty_group_is_an_error(self):
        with self.assertRaises(ValueError):
            estimate_object(np.zeros((3, 3)), [], 0)

    def test_invalid_center_method(self):
        with self.assertRaises(ValueError):
            estimate_object(np.zeros((1, 3)), [0], 0, center_method="median")


class BoxTest(unittest.TestCase):
    def test_corner_order(self):
        corners = box_corners(np.array([0, 0, 0]), np.array([1, 2, 3]))
        np.testing.assert_allclose(corners[:4, 2], 0)
        np.testing.assert_allclose(corners[4:, 2], 3)
        np.testing.assert_allclose(corners[4:, :2], corners[:4, :2])
        # Bottom face walks around the box: consecutive corners share an edge.
        edges = np.linalg.norm(
            corners[:4] - np.roll(corners[:4], -1, axis=0), axis=1
        )
        np.testing.assert_allclose(edges, [1, 2, 1, 2])

    def test_box_contains_all_points(self):
        rng = np.random.default_rng(0)
        points = rng.uniform(-5, 5, (200, 3))
        obj = estimate_object(points, np.arange(200), 0)
        self.assertTrue((points >= obj.corners.min(axis=0) - 1e-9).all())
        self.assertTrue((points <= obj.corners.max(axis=0) + 1e-9).all())


class TiltedSensorTest(unittest.TestCase):
    """A level object seen by the tilted dataset sensor."""

    def setUp(self):
        # 0.4 x 0.5 x 1.8 m person standing on the ground, in world.
        self.world = grid_box([408.7, -8.2, 0.0], [409.1, -7.7, 1.8], 7)
        self.cloud = DATASET_POSE.to_cloud(self.world)
        self.indices = np.arange(len(self.cloud))

    def test_pose_round_trip(self):
        np.testing.assert_allclose(
            DATASET_POSE.to_reference(self.cloud), self.world, atol=1e-9,
        )

    def test_size_is_level(self):
        obj = estimate_object(self.cloud, self.indices, 0, DATASET_POSE)
        np.testing.assert_allclose(obj.size, [0.4, 0.5, 1.8], atol=1e-9)

    def test_sensor_frame_box_would_be_inflated(self):
        # Why the box is not axis-aligned in the sensor frame.
        obj = estimate_object(self.cloud, self.indices, 0)
        self.assertGreater(obj.size[0], 0.4 + 0.5)

    def test_center_in_both_frames(self):
        obj = estimate_object(self.cloud, self.indices, 0, DATASET_POSE)
        np.testing.assert_allclose(
            obj.center_reference, [408.9, -7.95, 0.9], atol=1e-9,
        )
        np.testing.assert_allclose(
            DATASET_POSE.to_reference(obj.center[None])[0],
            [408.9, -7.95, 0.9], atol=1e-9,
        )

    def test_distances_from_sensor(self):
        obj = estimate_object(self.cloud, self.indices, 0, DATASET_POSE)
        sensor = np.array(DATASET_POSE.translation)
        offset = np.array([408.9, -7.95, 0.9]) - sensor
        self.assertAlmostEqual(obj.distance, np.linalg.norm(offset))
        self.assertAlmostEqual(
            obj.horizontal_distance, np.linalg.norm(offset[:2])
        )

    def test_corners_are_in_cloud_frame(self):
        obj = estimate_object(self.cloud, self.indices, 0, DATASET_POSE)
        world = DATASET_POSE.to_reference(obj.corners)
        np.testing.assert_allclose(world.min(axis=0), [408.7, -8.2, 0.0],
                                   atol=1e-9)
        np.testing.assert_allclose(world.max(axis=0), [409.1, -7.7, 1.8],
                                   atol=1e-9)

    def test_box_orientation_rotates_reference_axes_to_cloud(self):
        x, y, z, w = DATASET_POSE.box_orientation_xyzw
        # Reference z axis (up) in the cloud frame: third column of R(q).
        up = np.array([
            2 * (x * z + y * w),
            2 * (y * z - x * w),
            1 - 2 * (x * x + y * y),
        ])
        expected = DATASET_POSE.to_cloud(
            np.array([[0, 0, 1.0]]) + DATASET_POSE.translation
        )[0]
        np.testing.assert_allclose(up, expected, atol=1e-9)


class PoseTest(unittest.TestCase):
    def test_invalid(self):
        for args in (
            ((0, 0, 0, 2), (0, 0, 0)),
            ((0, 0, 0, 1), (0, 0, float("nan"))),
            ((0, 0, 1), (0, 0, 0)),
        ):
            with self.assertRaises(ValueError):
                ReferencePose(*args)


class FrameTest(unittest.TestCase):
    def setUp(self):
        self.points = np.vstack([
            grid_box([20, 0, 0], [21, 1, 1]),
            grid_box([5, 0, 0], [6, 1, 1]),
            grid_box([10, 0, 0], [11, 1, 1]),
        ])
        n = 125
        self.groups = [
            np.arange(0, n), np.arange(n, 2 * n), np.arange(2 * n, 3 * n),
        ]

    def test_ids_ascend_with_distance(self):
        objects, rejected = estimate_objects(self.points, self.groups)
        self.assertEqual(rejected, 0)
        self.assertEqual([o.id for o in objects], [0, 1, 2])
        distances = [o.distance for o in objects]
        self.assertEqual(distances, sorted(distances))
        # point_indices still select the object's own points.
        self.assertLess(self.points[objects[0].point_indices][:, 0].max(), 7)

    def test_no_groups(self):
        self.assertEqual(estimate_objects(np.empty((0, 3)), []), ([], 0))

    def test_footprint_limit(self):
        wall = grid_box([0, 30, 0], [20, 30.2, 3], 8)
        person = grid_box([5, 0, 0], [5.4, 0.4, 1.8])
        points = np.vstack([wall, person])
        groups = [np.arange(len(wall)), np.arange(len(wall), len(points))]

        objects, rejected = estimate_objects(
            points, groups, params=EstimatorParams(max_footprint=15.0),
        )
        self.assertEqual(rejected, 1)
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].point_count, len(person))

        objects, rejected = estimate_objects(
            points, groups, params=EstimatorParams(max_footprint=0.0),
        )
        self.assertEqual(rejected, 0)
        self.assertEqual(len(objects), 2)

    def test_height_limit(self):
        points = grid_box([0, 0, 0], [1, 1, 12])
        groups = [np.arange(len(points))]
        objects, rejected = estimate_objects(
            points, groups, params=EstimatorParams(max_height=10.0),
        )
        self.assertEqual((len(objects), rejected), (0, 1))

    def test_invalid_params(self):
        for kwargs in (
            {"center_method": "x"},
            {"max_footprint": -1.0},
            {"max_height": float("inf")},
        ):
            with self.assertRaises(ValueError):
                EstimatorParams(**kwargs)


class HeightBandTest(unittest.TestCase):
    def test_uses_reference_height(self):
        world = np.array([
            [420.0, 0.0, 0.5],
            [420.0, 0.0, 9.9],
            [420.0, 0.0, 10.1],
            [420.0, 0.0, -1.5],
        ])
        cloud = DATASET_POSE.to_cloud(world)
        np.testing.assert_array_equal(
            height_band_mask(cloud, DATASET_POSE, -1.0, 10.0),
            [True, True, False, False],
        )

    def test_empty(self):
        self.assertEqual(
            len(height_band_mask(np.empty((0, 3)), DATASET_POSE, 0, 1)), 0
        )

    def test_non_finite_is_excluded(self):
        points = np.array([[np.nan, 0, 0], [0, 0, 1.0]])
        pose = ReferencePose()
        np.testing.assert_array_equal(
            height_band_mask(points, pose, 0.0, 2.0), [False, True],
        )


if __name__ == "__main__":
    unittest.main()

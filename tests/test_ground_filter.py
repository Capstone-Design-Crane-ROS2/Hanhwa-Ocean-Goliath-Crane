"""Unit tests for tracking constrained RANSAC ground removal."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "lidar"),
)

from ground_filter import (
    GroundParams,
    GroundRemover,
    prior_from_transform,
)


# Dataset sensor pose: world <- OS1_A, z axis tilted 20 deg from vertical.
DATASET_Q = (0.17365, 0.0, 0.98481, 0.0)
DATASET_T = (396.8844, 0.052, 5.6691)


def quat_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def make_scene(seed=0, ground_points=6000, extra=None):
    """World-frame scene: ground at z=0, a 1.8 m person, a truck box.

    Returns (world_points, is_ground, is_object).
    """
    rng = np.random.default_rng(seed)
    ground = np.column_stack([
        rng.uniform(5, 40, ground_points),
        rng.uniform(-20, 20, ground_points),
        rng.normal(0.0, 0.005, ground_points),
    ])
    person = np.column_stack([
        rng.uniform(14.8, 15.2, 300),
        rng.uniform(-0.2, 0.2, 300),
        rng.uniform(0.1, 1.8, 300),
    ])
    truck = np.column_stack([
        rng.uniform(25, 30, 800),
        rng.uniform(5, 7.5, 800),
        rng.uniform(0.3, 2.3, 800),
    ])
    parts = [ground, person, truck] + ([extra] if extra is not None else [])
    labels = [np.full(len(ground), "g"), np.full(len(person), "o"),
              np.full(len(truck), "o")]
    if extra is not None:
        labels.append(np.full(len(extra), "x"))
    labels = np.concatenate(labels)
    return np.vstack(parts), labels == "g", labels == "o"


def to_sensor(world_points, q=DATASET_Q, t=DATASET_T):
    """Scene x/y are relative to the point below the sensor."""
    rot = quat_to_matrix(*q)
    return (world_points - [0.0, 0.0, t[2]]) @ rot


def platform(height, n=12000, seed=1):
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(5, 40, n), rng.uniform(-20, 20, n), np.full(n, height),
    ])


class TestPriorFromTransform(unittest.TestCase):

    def test_dataset_pose(self):
        normal, height = prior_from_transform(DATASET_Q, DATASET_T)

        np.testing.assert_allclose(
            normal, [0.342, 0.0, 0.9397], atol=1e-3,
        )
        self.assertAlmostEqual(height, 5.6691, places=4)

    def test_ground_z_offset(self):
        _, height = prior_from_transform(DATASET_Q, DATASET_T, 0.5)
        self.assertAlmostEqual(height, 5.1691, places=4)

    def test_invalid_transform(self):
        for q, t in (
            ((0, 0, 0, 0), DATASET_T),
            ((np.nan, 0, 0, 1), DATASET_T),
            ((0, 0, 0, 1), (0, np.inf, 0)),
            ((0, 0, 0, 2), DATASET_T),
        ):
            with self.assertRaises(ValueError):
                prior_from_transform(q, t)


class TestGroundParams(unittest.TestCase):

    def test_invalid_params(self):
        for kwargs in (
            dict(distance_threshold=0), dict(distance_threshold=np.nan),
            dict(iterations=0), dict(min_inlier_ratio=1.5),
            dict(max_hold_frames=-1), dict(min_inliers=2),
        ):
            with self.assertRaises(ValueError):
                GroundParams(**kwargs)


class TestGroundRemover(unittest.TestCase):

    def setUp(self):
        self.prior = prior_from_transform(DATASET_Q, DATASET_T)

    def run_scene(self, remover, world, prior=None):
        return remover.process(to_sensor(world), prior or self.prior)

    def test_tilted_sensor_removes_ground_keeps_objects(self):
        world, is_ground, is_object = make_scene()
        ground, result = self.run_scene(GroundRemover(), world)

        self.assertEqual(result.mode, "fit")
        self.assertEqual(ground.dtype, bool)
        self.assertEqual(len(ground), len(world))
        self.assertTrue(ground[is_ground].all())
        self.assertFalse(ground[is_object].any())
        self.assertAlmostEqual(result.height, 5.6691, delta=0.01)

    def test_elevated_platform_is_not_ground(self):
        # Platform larger than the ground; must not be chosen.
        for height in (1.5, 0.25):
            world, is_ground, _ = make_scene(extra=platform(height))
            ground, result = self.run_scene(GroundRemover(), world)

            self.assertEqual(result.mode, "fit")
            self.assertTrue(ground[is_ground].all())
            self.assertFalse(ground[~is_ground].any(), height)

    def test_distance_threshold(self):
        world, _, _ = make_scene()
        world = np.vstack([world, [[20.0, 0.0, 0.06]]])

        for threshold, expected in ((0.05, False), (0.08, True)):
            remover = GroundRemover(
                GroundParams(distance_threshold=threshold)
            )
            ground, _ = self.run_scene(remover, world)
            self.assertEqual(bool(ground[-1]), expected)

    def test_no_prior_is_passthrough(self):
        world, _, _ = make_scene()
        ground, result = GroundRemover().process(to_sensor(world), None)

        self.assertEqual(result.mode, "passthrough")
        self.assertFalse(ground.any())

    def test_failed_fit_does_not_reuse_previous_plane(self):
        remover = GroundRemover()
        world, _, _ = make_scene()
        _, first = self.run_scene(remover, world)
        self.assertEqual(first.mode, "fit")

        # Next frame: ground no longer visible, only low object points.
        _, is_ground, _ = make_scene()
        objects_only = world[~is_ground]
        low = np.column_stack([
            np.linspace(10, 30, 50), np.zeros(50), np.full(50, 0.04),
        ])
        frame = np.vstack([objects_only, low])
        ground, result = self.run_scene(remover, frame)

        self.assertEqual(result.mode, "passthrough")
        self.assertFalse(ground.any())

    def test_hold_only_when_enabled(self):
        remover = GroundRemover(GroundParams(max_hold_frames=1))
        world, is_ground, _ = make_scene()
        self.run_scene(remover, world)

        frame = np.vstack([
            world[~is_ground], [[20.0, 0.0, 0.0]],
        ])
        ground, result = self.run_scene(remover, frame)
        self.assertEqual(result.mode, "hold")
        self.assertTrue(ground[-1])

        _, result = self.run_scene(remover, frame)
        self.assertEqual(result.mode, "passthrough")

    def test_hold_rejected_when_prior_moved(self):
        remover = GroundRemover(GroundParams(max_hold_frames=5))
        world, is_ground, _ = make_scene()
        self.run_scene(remover, world)

        # Pose changed by 2 m: the old plane is outside prior bounds.
        normal, height = self.prior
        _, result = remover.process(
            to_sensor(world[~is_ground]), (normal, height + 2.0),
        )
        self.assertEqual(result.mode, "passthrough")

    def test_tracks_slow_drift(self):
        remover = GroundRemover()
        world, is_ground, is_object = make_scene()

        for step in range(1, 6):
            # Sensor pitch drifts 1 deg per frame (5 deg total).
            angle = np.radians(step)
            drift = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)],
            ])
            points = to_sensor(world) @ drift.T
            ground, result = remover.process(points, self.prior)

            self.assertEqual(result.mode, "fit", step)
            self.assertTrue(ground[is_ground].all(), step)
            self.assertFalse(ground[is_object].any(), step)

    def test_sudden_jump_beyond_prior_bounds_is_passthrough(self):
        remover = GroundRemover()
        world, _, _ = make_scene()
        shifted = world - [0.0, 0.0, 1.5]   # ground 1.5 m below prior

        ground, result = self.run_scene(remover, shifted)

        self.assertEqual(result.mode, "passthrough")
        self.assertFalse(ground.any())

    def test_empty_and_non_finite_input(self):
        ground, result = GroundRemover().process(
            np.empty((0, 3), dtype=np.float32), self.prior,
        )
        self.assertEqual(len(ground), 0)
        self.assertEqual(result.mode, "passthrough")

        world, _, _ = make_scene()
        points = to_sensor(world)
        points[:10] = np.nan
        ground, result = GroundRemover().process(points, self.prior)
        self.assertEqual(result.mode, "fit")
        self.assertFalse(ground[:10].any())

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            GroundRemover().process(np.zeros((3, 2)), self.prior)


if __name__ == "__main__":
    unittest.main()

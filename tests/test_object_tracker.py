"""Unit tests for frame-to-frame object tracking."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "lidar"),
)

from object_tracker import ObjectTracker, TrackerParams


NS = 1_000_000_000
PERIOD_NS = 83_333_333  # about 12 Hz, sensor time


def run(tracker, frames, start_ns=0, period_ns=PERIOD_NS, stamps=None):
    """Feeds (N, 3) position arrays; returns the per-frame ID lists."""
    out = []
    for k, positions in enumerate(frames):
        stamp = stamps[k] if stamps is not None else start_ns + k * period_ns
        out.append(tracker.update(stamp, positions, "world"))
    return out


class ParamsTest(unittest.TestCase):
    def test_rejects_invalid(self):
        with self.assertRaises(ValueError):
            TrackerParams(min_hits=0)
        with self.assertRaises(ValueError):
            TrackerParams(accel_std=0.0)
        with self.assertRaises(ValueError):
            TrackerParams(max_coast_s=-1.0)
        with self.assertRaises(ValueError):
            TrackerParams(meas_std_xy=float("nan"))


class LifecycleTest(unittest.TestCase):
    def test_confirmed_after_min_hits(self):
        tracker = ObjectTracker(TrackerParams(min_hits=3))
        ids = run(tracker, [[[10.0, 0.0, 1.0]]] * 4)
        self.assertEqual(ids[0], [None])
        self.assertEqual(ids[1], [None])
        self.assertEqual(ids[2], [0])
        self.assertEqual(ids[3], [0])

    def test_min_hits_one_gives_id_immediately(self):
        tracker = ObjectTracker(TrackerParams(min_hits=1))
        self.assertEqual(tracker.update(0, [[1.0, 2.0, 0.0]]), [0])

    def test_empty_frames(self):
        tracker = ObjectTracker()
        self.assertEqual(tracker.update(0, np.empty((0, 3))), [])
        self.assertEqual(tracker.update(PERIOD_NS, []), [])

    def test_static_objects_keep_ids_with_noise(self):
        rng = np.random.default_rng(0)
        truth = np.array([[10.0, 0.0, 1.0], [12.0, 3.0, 1.0],
                          [30.0, -5.0, 2.0]])
        frames = [truth + rng.normal(0, 0.1, truth.shape)
                  for _ in range(100)]
        # Detection order changes every frame, as frame-local IDs do.
        frames = [f[rng.permutation(3)] for f in frames]
        tracker = ObjectTracker()
        ids = run(tracker, frames)
        by_position = {}
        for positions, frame_ids in zip(frames[2:], ids[2:]):
            for p, tid in zip(positions, frame_ids):
                key = int(np.argmin(np.linalg.norm(truth - p, axis=1)))
                by_position.setdefault(key, set()).add(tid)
        self.assertEqual({k: len(v) for k, v in by_position.items()},
                         {0: 1, 1: 1, 2: 1})
        self.assertNotIn(None, set().union(*by_position.values()))

    def test_ids_never_reused(self):
        tracker = ObjectTracker(TrackerParams(min_hits=1, max_coast_s=0.1))
        first = tracker.update(0, [[0.0, 0.0, 0.0]])
        tracker.update(NS, [])            # deleted after coasting
        second = tracker.update(NS + PERIOD_NS, [[0.0, 0.0, 0.0]])
        self.assertNotEqual(first, second)


class MotionTest(unittest.TestCase):
    def test_moving_object_and_velocity(self):
        # 2 m/s crossing, like the S7 truck.
        tracker = ObjectTracker()
        frames = [[[20.0, -10.0 + 2.0 * k * PERIOD_NS / NS, 1.0]]
                  for k in range(60)]
        ids = run(tracker, frames)
        self.assertEqual({i[0] for i in ids[2:]}, {0})
        np.testing.assert_allclose(tracker.tracks[0].velocity,
                                   [0.0, 2.0, 0.0], atol=0.05)

    def test_uses_real_stamp_difference(self):
        # Irregular stamps: a fixed-period model would mispredict and lose
        # the fast object, the stamp-based one keeps it and its speed.
        rng = np.random.default_rng(1)
        steps = rng.uniform(0.03, 0.4, 80)
        stamps = (np.concatenate([[0.0], np.cumsum(steps)]) * NS).astype(int)
        speed = 5.0
        frames = [[[0.0, speed * s / NS, 0.0]] for s in stamps]
        tracker = ObjectTracker()
        ids = run(tracker, frames, stamps=stamps)
        self.assertEqual({i[0] for i in ids[2:]}, {0})
        self.assertAlmostEqual(tracker.tracks[0].velocity[1], speed,
                               delta=0.1)

    def test_crossing_objects_keep_identity(self):
        # Two objects passing 1.2 m apart in opposite directions.
        tracker = ObjectTracker()
        frames = []
        for k in range(80):
            t = k * PERIOD_NS / NS
            frames.append([[-8.0 + 2.0 * t, 0.0, 1.0],
                           [8.0 - 2.0 * t, 1.2, 1.0]])
        ids = run(tracker, frames)
        self.assertEqual({i[0] for i in ids[2:]}, {0})
        self.assertEqual({i[1] for i in ids[2:]}, {1})


class CoastingTest(unittest.TestCase):
    def test_reattaches_after_short_miss(self):
        # Moving object missing for 0.8 s (< max_coast_s): same ID back at
        # the predicted position.
        params = TrackerParams(max_coast_s=1.5)
        tracker = ObjectTracker(params)
        stamps, frames = [], []
        for k in range(50):
            t = k * PERIOD_NS / NS
            stamps.append(k * PERIOD_NS)
            hidden = 20 <= k < 30
            frames.append([] if hidden else [[1.5 * t, 0.0, 0.0]])
        ids = run(tracker, frames, stamps=stamps)
        seen = {i[0] for i in ids[2:] if i}
        self.assertEqual(seen, {0})
        self.assertEqual(ids[30], [0])

    def test_deleted_after_max_coast(self):
        tracker = ObjectTracker(TrackerParams(max_coast_s=0.5))
        run(tracker, [[[5.0, 0.0, 0.0]]] * 5)
        self.assertIn(0, tracker.tracks)
        tracker.update(5 * PERIOD_NS + int(0.6 * NS), [], "world")
        self.assertNotIn(0, tracker.tracks)

    def test_tentative_dropped_quickly(self):
        tracker = ObjectTracker(TrackerParams(tentative_coast_s=0.1))
        tracker.update(0, [[5.0, 0.0, 0.0]], "world")
        tracker.update(int(0.2 * NS), [], "world")
        self.assertEqual(tracker.tracks, {})

    def test_far_detection_not_attached(self):
        tracker = ObjectTracker(TrackerParams(max_gate_distance=3.0))
        run(tracker, [[[0.0, 0.0, 0.0]]] * 5)
        ids = tracker.update(5 * PERIOD_NS, [[0.0, 5.0, 0.0]], "world")
        self.assertEqual(ids, [None])        # new tentative track
        self.assertEqual(len(tracker.tracks), 2)


class ResetTest(unittest.TestCase):
    def test_stamp_going_back_resets(self):
        tracker = ObjectTracker()
        run(tracker, [[[0.0, 0.0, 0.0]]] * 5, start_ns=10 * NS)
        tracker.update(NS, [[0.0, 0.0, 0.0]], "world")
        self.assertEqual(tracker.resets, 1)
        self.assertEqual(list(tracker.tracks), [1])

    def test_large_gap_resets(self):
        tracker = ObjectTracker(TrackerParams(max_gap_s=2.0))
        run(tracker, [[[0.0, 0.0, 0.0]]] * 5)
        tracker.update(10 * NS, [[0.0, 0.0, 0.0]], "world")
        self.assertEqual(tracker.resets, 1)

    def test_reference_frame_change_resets(self):
        tracker = ObjectTracker()
        run(tracker, [[[0.0, 0.0, 0.0]]] * 5)
        tracker.update(5 * PERIOD_NS, [[0.0, 0.0, 0.0]], "OS1_A")
        self.assertEqual(tracker.resets, 1)

    def test_duplicate_stamp_does_not_predict(self):
        tracker = ObjectTracker()
        run(tracker, [[[0.0, 0.0, 0.0]]] * 3)
        before = tracker.tracks[0].P.copy()
        tracker.update(2 * PERIOD_NS, [], "world")
        np.testing.assert_array_equal(tracker.tracks[0].P, before)


class StateAccessTest(unittest.TestCase):
    """Read-only state used by object_motion.py; IDs are unaffected."""

    def test_last_tracks_follow_detections(self):
        tracker = ObjectTracker()
        frames = [[[10.0, 0.0, 1.0], [0.0, 10.0, 1.0]]] * 4
        ids = run(tracker, frames)[-1]
        self.assertEqual(len(tracker.last_tracks), 2)
        for tid, track in zip(ids, tracker.last_tracks):
            self.assertEqual(track.id, tid)
        # Swapped order: the same tracks, in the new detection order.
        swapped = tracker.update(4 * PERIOD_NS,
                                 [[0.0, 10.0, 1.0], [10.0, 0.0, 1.0]],
                                 "world")
        self.assertEqual(swapped, ids[::-1])
        self.assertEqual([t.id for t in tracker.last_tracks], ids[::-1])

    def test_last_tracks_include_tentative(self):
        tracker = ObjectTracker()
        ids = tracker.update(0, [[1.0, 0.0, 0.0]], "world")
        self.assertEqual(ids, [None])
        self.assertFalse(tracker.last_tracks[0].confirmed)
        self.assertEqual(tracker.update(PERIOD_NS, [], "world"), [])
        self.assertEqual(tracker.last_tracks, [])

    def test_ids_unchanged_by_state_access(self):
        a, b = ObjectTracker(), ObjectTracker()
        frames = [[[k * 0.1, 0.0, 1.0], [5.0, k * 0.2, 1.0]]
                  for k in range(10)]
        ids_a = run(a, frames)
        ids_b = []
        for k, positions in enumerate(frames):
            ids_b.append(b.update(k * PERIOD_NS, positions, "world"))
            b.coasting_tracks()
            [t.velocity_covariance for t in b.last_tracks if t]
        self.assertEqual(ids_a, ids_b)

    def test_coasting_tracks(self):
        tracker = ObjectTracker()
        run(tracker, [[[10.0, 0.0, 1.0]]] * 4)
        self.assertEqual(tracker.coasting_tracks(), [])
        tracker.update(4 * PERIOD_NS, [], "world")
        coasting = tracker.coasting_tracks()
        self.assertEqual([t.id for t in coasting], [0])
        self.assertAlmostEqual(coasting[0].since_update_s(4 * PERIOD_NS),
                               PERIOD_NS * 1e-9)

    def test_velocity_covariance_and_age(self):
        tracker = ObjectTracker()
        run(tracker, [[[k * 0.2, 0.0, 1.0]] for k in range(12)])
        track = tracker.tracks[0]
        cov = track.velocity_covariance
        self.assertEqual(cov.shape, (3, 3))
        np.testing.assert_allclose(cov, cov.T)
        self.assertTrue(np.all(np.linalg.eigvalsh(cov) > 0))
        self.assertLess(cov[0, 0], TrackerParams().init_vel_std ** 2)
        self.assertAlmostEqual(track.age_s(11 * PERIOD_NS),
                               11 * PERIOD_NS * 1e-9)

    def test_nis_reflects_innovation(self):
        tracker = ObjectTracker()
        run(tracker, [[[0.0, 0.0, 1.0]]] * 10)
        small = tracker.tracks[0].nis
        tracker.update(10 * PERIOD_NS, [[0.8, 0.0, 1.0]], "world")
        self.assertGreater(tracker.tracks[0].nis, small)


if __name__ == "__main__":
    unittest.main()

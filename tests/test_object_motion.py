"""Velocity validity and crane-boundary TTC on synthetic trajectories.

Boundary: the plane x = 0, objects come from +x (outward normal +x), so an
object approaches the crane when its x decreases. Stamps are sensor time,
by default the S7 GT interval mix (66.7 ms 74 %, 133.3 ms 20 %, 200 ms 5 %,
266.7 ms 1 %, measured from /eval/ground_truth).

After a long gap the Kalman velocity sigma grows and the velocity can be
withheld for a frame (velocity_unknown). That is intended, so most checks
are: never a wrong status, and the expected status in most settled frames.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lidar"))

from crane_boundary import (  # noqa: E402
    CraneState,
    ExclusionZone,
    PlaneBoundary,
    zones_from_flat,
)
from object_motion import (  # noqa: E402
    MotionEstimator,
    MotionParams,
    TtcParams,
    assess_boundary,
    model_info,
    motion_to_dict,
    risk_to_dict,
)
from object_tracker import ObjectTracker  # noqa: E402


NS = 1_000_000_000
PLANE = PlaneBoundary((0.0, 0.0), (1.0, 0.0))
S7_GAP_MIX = ([66.7, 133.3, 200.0, 266.7], [0.74, 0.20, 0.05, 0.01])


def s7_gaps(count, seed=0):
    rng = np.random.default_rng(seed)
    return list(rng.choice(S7_GAP_MIX[0], size=count, p=S7_GAP_MIX[1]))


def stamps(count, gaps_ms=None, start_ns=10 * NS):
    gaps_ms = gaps_ms or s7_gaps(count)
    out, t = [], start_ns
    for k in range(count):
        out.append(t)
        t += int(gaps_ms[k % len(gaps_ms)] * 1e6)
    return out


def box(center, size, point_count=200):
    center = np.asarray(center, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    return SimpleNamespace(
        center_reference=center, size=size, point_count=point_count,
        bbox_min=center - size / 2, bbox_max=center + size / 2,
    )


class Runner:
    """Feeds frames of boxes through tracker, motion and boundary."""

    def __init__(self, motion_params=None, ttc_params=None, zones=(),
                 crane=None, seed=0):
        self.tracker = ObjectTracker()
        self.motion = MotionEstimator(motion_params)
        self.ttc = ttc_params or TtcParams()
        self.zones = zones
        self.crane = crane
        self.rng = np.random.default_rng(seed)

    def step(self, stamp, objects, noise=0.03, world=True):
        noisy = []
        for obj in objects:
            jitter = np.zeros(3)
            jitter[:2] = self.rng.normal(0, noise, 2)
            noisy.append(box(obj.center_reference + jitter, obj.size,
                             obj.point_count))
        ids = self.tracker.update(
            stamp, [o.center_reference for o in noisy], "world")
        motions = self.motion.update(self.tracker, stamp, noisy, world)
        risks = [
            assess_boundary(m, o.bbox_min, o.bbox_max, o.center_reference,
                            PLANE, self.ttc, self.crane, self.zones, world)
            for o, m in zip(noisy, motions)
        ]
        return ids, motions, risks


def trajectory(start, velocity, times_ns, size=(2.0, 2.0, 2.0)):
    t0 = times_ns[0]
    return [box(np.asarray(start) + np.asarray(velocity)
                * (t - t0) * 1e-9, size) for t in times_ns]


def run_single(runner, boxes, times_ns, **kwargs):
    out = []
    for stamp, obj in zip(times_ns, boxes):
        ids, motions, risks = runner.step(stamp, [obj], **kwargs)
        out.append((ids[0], motions[0], risks[0], obj))
    return out


class ParamsTest(unittest.TestCase):
    def test_rejects_invalid(self):
        with self.assertRaises(ValueError):
            MotionParams(max_speed_std=0.0)
        with self.assertRaises(ValueError):
            MotionParams(stationary_speed=-0.1)
        with self.assertRaises(ValueError):
            MotionParams(min_hits=1.5)
        with self.assertRaises(ValueError):
            TtcParams(max_ttc_s=0.0)
        with self.assertRaises(ValueError):
            TtcParams(safety_margin_m=float("nan"))


class BoundaryTest(unittest.TestCase):
    def test_distance_uses_nearest_box_point(self):
        # Center 10 m out, 4 m deep: nearest face 8 m out.
        self.assertAlmostEqual(
            PLANE.distance((8.0, -1.0, 0.0), (12.0, 1.0, 2.0)), 8.0)

    def test_signed_inside(self):
        self.assertAlmostEqual(
            PLANE.distance((-1.0, 0.0, 0.0), (1.0, 1.0, 1.0)), -1.0)

    def test_rotated_normal(self):
        plane = PlaneBoundary((0.0, 0.0), (0.0, 2.0))  # normalized
        np.testing.assert_allclose(plane.normal(), [0.0, 1.0])
        self.assertAlmostEqual(
            plane.distance((-5.0, 3.0, 0.0), (5.0, 4.0, 1.0)), 3.0)

    def test_dataset_face(self):
        face = PlaneBoundary((398.884, 0.052), (1.0, 0.0))
        # S7 truck GT box at eval x = 20 (2.45 m wide in x).
        self.assertAlmostEqual(
            face.distance((417.657, -29.4, 0.0), (420.111, -24.5, 2.3)),
            18.773, places=3)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            PlaneBoundary((0.0, 0.0), (0.0, 0.0))
        with self.assertRaises(ValueError):
            ExclusionZone((1, 0, 0), (0, 1, 1))
        with self.assertRaises(ValueError):
            zones_from_flat([0, 0, 0, 1, 1])

    def test_zones_from_flat(self):
        zones = zones_from_flat([0, 0, 0, 1, 1, 1, 5, 5, 5, 6, 6, 6])
        self.assertEqual(len(zones), 2)
        self.assertTrue(zones[1].contains((5.5, 5.5, 5.5)))
        self.assertFalse(zones[0].contains((1.5, 0.5, 0.5)))


def ttc_tolerance(true_ttc):
    """10 %, but at least 0.15 s (the last metres before the face)."""
    return max(0.1 * true_ttc, 0.15)


class ScenarioTest(unittest.TestCase):
    """The ten required situations (numbers as in the task)."""

    def settled(self, results, after_s=1.5):
        return [r for r in results
                if r[1].track_age_s is not None
                and r[1].track_age_s >= after_s]

    def assert_status(self, results, expected, share=0.8):
        """Every frame expected or withheld; expected in >= share."""
        self.assertTrue(results)
        statuses = [r[2].status for r in results]
        self.assertTrue(set(statuses) <= {expected, "velocity_unknown"},
                        set(statuses))
        self.assertGreaterEqual(statuses.count(expected) / len(statuses),
                                share)

    def test_1_constant_approach_ttc(self):
        times = stamps(120)
        boxes = trajectory((25.0, 0.0, 1.0), (-1.5, 0.0, 0.0), times)
        results = run_single(Runner(), boxes, times)
        valid = [r for r in self.settled(results) if r[2].status == "valid"]
        self.assertGreater(len(valid), 30)
        self.assert_status(
            [r for r in self.settled(results) if r[3].bbox_min[0] < 15],
            "valid")
        for _, motion, risk, obj in valid:
            true_ttc = (obj.bbox_min[0]) / 1.5
            self.assertEqual(risk.direction, "approaching")
            self.assertAlmostEqual(risk.ttc, true_ttc,
                                   delta=ttc_tolerance(true_ttc))
            low, high = risk.ttc_range
            self.assertLessEqual(low, risk.ttc)
            self.assertGreaterEqual(high, risk.ttc)
            self.assertEqual(motion.state, "moving")
        # Distance is to the nearest box face, not the center.
        _, _, risk, obj = valid[-1]
        self.assertAlmostEqual(risk.distance, obj.bbox_min[0], delta=0.15)

    def test_1_beyond_range_far_away(self):
        times = stamps(40)
        boxes = trajectory((80.0, 0.0, 1.0), (-1.5, 0.0, 0.0), times)
        results = self.settled(run_single(Runner(), boxes, times))
        self.assert_status(results, "beyond_range")
        for _, _, risk, _ in results:
            self.assertIsNone(risk.ttc)

    def test_2_receding(self):
        times = stamps(60)
        boxes = trajectory((5.0, 0.0, 1.0), (1.5, 0.0, 0.0), times)
        results = self.settled(run_single(Runner(), boxes, times))
        self.assert_status(results, "not_approaching")
        for _, _, risk, _ in results:
            self.assertIn(risk.direction, ("receding", "unknown"))
            self.assertIsNone(risk.ttc)

    def test_3_parallel(self):
        # Like the S7 truck: constant x, moving along the face.
        times = stamps(80)
        boxes = trajectory((20.0, -27.0, 1.0), (0.0, 2.0, 0.0), times,
                           (2.45, 4.9, 2.3))
        results = self.settled(run_single(Runner(), boxes, times))
        self.assert_status(results, "not_approaching")
        for _, motion, risk, _ in results:
            self.assertIn(motion.state, ("moving", "unknown"))
            self.assertIn(risk.direction, ("steady", "unknown"))
            self.assertIsNone(risk.ttc)

    def test_4_stationary(self):
        times = stamps(60)
        boxes = trajectory((6.0, 3.0, 1.0), (0.0, 0.0, 0.0), times)
        results = self.settled(run_single(Runner(), boxes, times))
        self.assert_status(results, "not_approaching")
        for _, motion, risk, _ in results:
            self.assertIn(motion.state, ("stationary", "unknown"))
            self.assertIsNone(risk.ttc)

    def test_5_near_zero_closing(self):
        times = stamps(80)
        boxes = trajectory((10.0, 0.0, 1.0), (-0.1, 1.0, 0.0), times)
        results = self.settled(run_single(Runner(), boxes, times))
        self.assert_status(results, "not_approaching")
        for _, _, risk, _ in results:
            self.assertIn(risk.direction, ("steady", "unknown"))
            self.assertIsNone(risk.ttc)

    def test_6_inside_and_margin(self):
        times = stamps(30)
        inside = trajectory((0.5, 0.0, 1.0), (0.0, 0.0, 0.0), times)
        results = run_single(Runner(), inside, times)
        self.assertTrue(all(r[2].status == "inside_boundary"
                            for r in results if r[0] is not None))
        self.assertLess(results[-1][2].distance, 0)

        margin = Runner(ttc_params=TtcParams(safety_margin_m=2.0))
        near = trajectory((2.5, 0.0, 1.0), (0.0, 0.0, 0.0), times)
        results = run_single(margin, near, times)
        self.assertEqual(results[-1][2].status, "within_margin")
        self.assertIsNone(results[-1][2].ttc)

    def test_6_margin_shortens_ttc(self):
        times = stamps(60, [66.7])
        boxes = trajectory((20.0, 0.0, 1.0), (-2.0, 0.0, 0.0), times)
        plain = run_single(Runner(seed=1), boxes, times)
        margin = run_single(
            Runner(ttc_params=TtcParams(safety_margin_m=2.0), seed=1),
            boxes, times)
        a, b = plain[-1][2], margin[-1][2]
        self.assertEqual(a.status, "valid")
        self.assertEqual(b.status, "valid")
        self.assertAlmostEqual(a.ttc - b.ttc, 2.0 / a.closing_speed,
                               places=6)

    def test_7_fragmenting_static_object_never_gets_ttc(self):
        # Static truck seen as one box or as its front fragment, at random
        # (S4 yaw 0, 30 m): the center jumps 1.2 m toward the crane.
        rng = np.random.default_rng(3)
        times = stamps(150)
        runner = Runner()
        statuses, states = [], []
        for stamp in times:
            if rng.random() < 0.3:
                obj = box((10.6, 0.0, 1.0), (2.4, 2.0, 2.0), 90)
            else:
                obj = box((11.8, 0.0, 1.0), (4.9, 2.0, 2.0), 200)
            ids, motions, risks = runner.step(stamp, [obj])
            statuses.append(risks[0].status)
            states.append((motions[0].state, motions[0].reason))
        self.assertNotIn("valid", statuses)
        self.assertNotIn("moving", [s for s, _ in states])
        self.assertIn(("unknown", "shape_jump"), states)

    def test_7_initial_and_uncertain_velocity_withheld(self):
        times = stamps(30)
        boxes = trajectory((15.0, 0.0, 1.0), (-2.0, 0.0, 0.0), times)
        results = run_single(Runner(), boxes, times)
        first = [r for r in results if r[1].track_id is not None][0]
        self.assertEqual(first[1].state, "unknown")
        self.assertEqual(first[1].reason, "initializing")
        self.assertEqual(first[2].status, "velocity_unknown")
        self.assertIsNone(first[2].ttc)
        self.assertIsNone(first[2].closing_speed)
        # Unconfirmed track: no motion at all.
        self.assertEqual(results[0][1].reason, "unconfirmed_track")
        self.assertIsNone(results[0][1].velocity)
        self.assertIsNone(results[0][1].track_id)

    def test_7_tight_uncertainty_limit(self):
        times = stamps(40)
        boxes = trajectory((15.0, 0.0, 1.0), (-2.0, 0.0, 0.0), times)
        runner = Runner(MotionParams(max_speed_std=0.05))
        for _, motion, risk, _ in run_single(runner, boxes, times)[3:]:
            self.assertEqual(motion.state, "unknown")
            self.assertIn(motion.reason, ("initializing",
                                          "high_uncertainty"))
            self.assertEqual(risk.status, "velocity_unknown")

    def test_8_reset_restarts_validation(self):
        times = stamps(40, [66.7])
        boxes = trajectory((15.0, 0.0, 1.0), (-1.5, 0.0, 0.0), times)
        runner = Runner()
        results = run_single(runner, boxes, times)
        self.assertEqual(results[-1][2].status, "valid")
        old_id = results[-1][0]
        # Bag restart: stamp goes back, all tracks dropped.
        again = run_single(runner, boxes[:6], times[:6])
        self.assertEqual(runner.tracker.resets, 1)
        confirmed = [r for r in again if r[0] is not None]
        self.assertTrue(confirmed)
        self.assertNotEqual(confirmed[0][0], old_id)
        for _, motion, risk, _ in confirmed:
            self.assertEqual(motion.reason, "initializing")
            self.assertEqual(risk.status, "velocity_unknown")

    def test_8_new_track_after_loss_starts_over(self):
        times = stamps(60, [66.7])
        boxes = trajectory((25.0, 0.0, 1.0), (-1.5, 0.0, 0.0), times)
        runner = Runner()
        first = run_single(runner, boxes[:30], times[:30])
        # 2 s without detection: the track is deleted (max_coast_s 1.5).
        resumed = [t + 2 * NS for t in times[30:]]
        later = run_single(
            runner, trajectory((10.0, 0.0, 1.0), (-1.5, 0.0, 0.0), resumed),
            resumed)
        confirmed = [r for r in later if r[0] is not None]
        self.assertNotEqual(confirmed[0][0], first[-1][0])
        self.assertEqual(confirmed[0][1].reason, "initializing")

    def test_9_drops_and_irregular_stamps(self):
        rng = np.random.default_rng(7)
        gaps = rng.choice([66.7, 133.3, 200.0, 266.7], size=200,
                          p=[0.6, 0.25, 0.1, 0.05])
        times = stamps(200, list(gaps))
        keep = rng.random(len(times)) > 0.3
        times = [t for t, k in zip(times, keep) if k]
        boxes = trajectory((40.0, 5.0, 1.0), (-2.0, -0.5, 0.0), times)
        results = run_single(Runner(), boxes, times)
        valid = [r for r in self.settled(results, 2.0)
                 if r[2].status == "valid"]
        self.assertGreater(len(valid), 10)
        for _, motion, risk, obj in valid:
            true_ttc = obj.bbox_min[0] / 2.0
            self.assertAlmostEqual(risk.ttc, true_ttc,
                                   delta=ttc_tolerance(true_ttc))
            self.assertLess(np.linalg.norm(motion.velocity - [-2.0, -0.5]),
                            0.3)

    def test_10_crane_structure_excluded(self):
        # The dataset's crane part: 4.7 m behind the face, 6 m high.
        zones = [ExclusionZone((-10.0, -35.0, 4.5), (0.0, 35.0, 12.0))]
        times = stamps(60)
        part = trajectory((-4.7, 0.1, 6.0), (0.0, 0.0, 0.0), times,
                          (2.6, 4.1, 1.1))
        runner = Runner(zones=zones)
        for _, _, risk, _ in run_single(runner, part, times, noise=0.1):
            self.assertEqual(risk.status, "excluded")
        # Without the zone it is inside the boundary, still never a TTC.
        for _, _, risk, _ in run_single(Runner(), part, times, noise=0.1):
            self.assertIn(risk.status, ("inside_boundary",))
            self.assertIsNone(risk.ttc)

    def test_10_person_under_the_crane_not_excluded(self):
        zones = [ExclusionZone((-10.0, -35.0, 4.5), (0.0, 35.0, 12.0))]
        times = stamps(20)
        person = trajectory((-2.0, 0.0, 0.9), (0.0, 0.0, 0.0), times,
                            (0.4, 0.5, 1.8))
        results = run_single(Runner(zones=zones), person, times)
        self.assertEqual(results[-1][2].status, "inside_boundary")


class OtherCasesTest(unittest.TestCase):
    def test_not_world_frame(self):
        times = stamps(20)
        boxes = trajectory((10.0, 0.0, 1.0), (-1.0, 0.0, 0.0), times)
        runner = Runner()
        for stamp, obj in zip(times, boxes):
            _, motions, risks = runner.step(stamp, [obj], world=False)
        self.assertEqual(risks[0].status, "not_world_frame")
        self.assertIsNone(risks[0].distance)
        self.assertIn(motions[0].reason,
                      ("not_world_frame", "unconfirmed_track"))

    def test_crane_velocity_is_relative(self):
        # Object static, crane moving toward it at 1 m/s (+x): closing 1.
        times = stamps(60)
        boxes = trajectory((15.0, 0.0, 1.0), (0.0, 0.0, 0.0), times)
        runner = Runner(crane=CraneState((1.0, 0.0)))
        _, motion, risk, _ = run_single(runner, boxes, times)[-1]
        self.assertEqual(motion.state, "stationary")
        self.assertAlmostEqual(risk.closing_speed, 1.0, delta=0.1)
        # A stationary object gets no TTC even then: the first model only
        # rates moving objects (documented limitation).
        self.assertEqual(risk.status, "not_approaching")

    def test_coasting_track_is_marked_predicted(self):
        times = stamps(40, [66.7])
        boxes = trajectory((20.0, 0.0, 1.0), (-1.5, 0.0, 0.0), times)
        runner = Runner()
        run_single(runner, boxes[:30], times[:30])
        runner.tracker.update(times[30], [], "world")
        runner.motion.update(runner.tracker, times[30], [])
        predicted = runner.motion.predicted(runner.tracker, times[30])
        self.assertEqual(len(predicted), 1)
        track, motion = predicted[0]
        self.assertFalse(motion.measured)
        self.assertGreater(motion.since_update_s, 0)
        self.assertEqual(motion.state, "moving")

    def test_window_mismatch_marks_inconsistent(self):
        times = stamps(40, [66.7])
        boxes = trajectory((20.0, 0.0, 1.0), (-1.5, 0.0, 0.0), times)
        runner = Runner(MotionParams(max_window_mismatch=0.0))
        _, motion, risk, _ = run_single(runner, boxes, times)[-1]
        self.assertEqual(motion.reason, "inconsistent")
        self.assertIsNotNone(motion.window_velocity)
        self.assertEqual(risk.status, "velocity_unknown")

    def test_implausible_speed(self):
        times = stamps(40, [66.7])
        boxes = trajectory((20.0, 0.0, 1.0), (-3.0, 0.0, 0.0), times)
        runner = Runner(MotionParams(max_speed=2.0))
        _, motion, _, _ = run_single(runner, boxes, times)[-1]
        self.assertEqual(motion.reason, "implausible_speed")

    def test_objects_must_match_tracker(self):
        tracker = ObjectTracker()
        tracker.update(0, [[1.0, 0.0, 0.0]], "world")
        with self.assertRaises(ValueError):
            MotionEstimator().update(tracker, 0, [])


class DictTest(unittest.TestCase):
    def test_unknown_has_no_velocity_or_ttc(self):
        times = stamps(4)
        boxes = trajectory((10.0, 0.0, 1.0), (-1.0, 0.0, 0.0), times)
        _, motion, risk, _ = run_single(Runner(), boxes, times)[-1]
        m, r = motion_to_dict(motion), risk_to_dict(risk)
        self.assertFalse(m["valid"])
        self.assertIsNone(m["velocity_mps"])
        self.assertIsNone(m["speed_mps"])
        self.assertIsNotNone(m["raw_velocity_mps"])
        self.assertIsNone(r["ttc_s"])
        self.assertIsNone(r["ttc_range_s"])
        self.assertEqual(r["ttc_status"], "velocity_unknown")

    def test_valid_dict(self):
        times = stamps(60, [66.7])
        boxes = trajectory((15.0, 0.0, 1.0), (-1.5, 0.0, 0.0), times)
        _, motion, risk, _ = run_single(Runner(), boxes, times)[-1]
        m, r = motion_to_dict(motion), risk_to_dict(risk)
        self.assertEqual(m["state"], "moving")
        self.assertEqual(len(m["velocity_mps"]), 2)
        self.assertEqual(r["ttc_status"], "valid")
        self.assertGreater(r["ttc_s"], 0)
        self.assertEqual(len(r["ttc_range_s"]), 2)

    def test_model_info(self):
        info = model_info(PLANE, MotionParams(), TtcParams(),
                          zones=zones_from_flat([0, 0, 0, 1, 1, 1]))
        model = info["crane_boundary_model"]
        self.assertEqual(model["boundary"]["type"], "plane")
        self.assertEqual(model["crane_motion"], "static_assumed")
        self.assertIn("not a collision prediction", model["note"])
        self.assertEqual(len(model["exclusion_zones"]), 1)
        self.assertEqual(info["motion_model"]["reference_frame"], "world")


if __name__ == "__main__":
    unittest.main()

"""Experimental validity variants of tools/exp_motion_validity.py."""

import sys
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "lidar"))
sys.path.insert(0, str(root / "tools"))

try:
    from exp_motion_validity import TrendMotion, transient_mixin
    from object_motion import MotionEstimator, MotionParams
    from object_tracker import ObjectTracker
except ImportError:  # pragma: no cover - ROS not sourced
    TrendMotion = None

S = 1_000_000_000


def history(sizes, dt=0.1):
    samples = deque((int(k * dt * S), np.zeros(2), np.asarray(s, float))
                    for k, s in enumerate(sizes))
    return SimpleNamespace(samples=samples)


@unittest.skipIf(TrendMotion is None, "ROS 2 Python not available")
class ShapeTest(unittest.TestCase):
    def test_gait_oscillation_passes_trend_only(self):
        # Footprint length 0.2 <-> 0.75 m (stride), 1 s of samples.
        sizes = [[0.2 if k % 4 < 2 else 0.75, 0.5] for k in range(11)]
        h = history(sizes)
        self.assertTrue(MotionEstimator()._shape_changing(h))
        self.assertFalse(TrendMotion()._shape_changing(h))

    def test_monotonic_growth_is_held_by_both(self):
        # Truck emerging next to a wall: 2.2 -> 4.9 m in 1 s (1st S7).
        sizes = [[2.4, 2.2 + 0.27 * k] for k in range(11)]
        h = history(sizes)
        self.assertTrue(MotionEstimator()._shape_changing(h))
        self.assertTrue(TrendMotion()._shape_changing(h))


@unittest.skipIf(TrendMotion is None, "ROS 2 Python not available")
class TransientTest(unittest.TestCase):
    def run_stop(self, estimator):
        """1.2 m/s for 4 s, then stopped; states per frame after the stop."""
        tracker = ObjectTracker()
        states = []
        for k in range(70):
            t = k * S // 10
            x = 1.2 * min(k, 40) * 0.1
            obj = SimpleNamespace(center_reference=np.array([x, 0.0, 0.9]),
                                  size=np.array([0.4, 0.5, 1.7]),
                                  point_count=40)
            tracker.update(t, [obj.center_reference], "world")
            m = estimator.update(tracker, t, [obj])[0]
            if k > 40:
                states.append((m.state, m.reason))
        return states

    def test_stop_moving_released_sooner(self):
        base = self.run_stop(MotionEstimator())
        trans = self.run_stop(transient_mixin(MotionEstimator, 0.4, 0.3, 4)(
            MotionParams()))
        first_not_moving = lambda s: next(  # noqa: E731
            k for k, (state, _) in enumerate(s) if state != "moving")
        self.assertLess(first_not_moving(trans), first_not_moving(base))
        self.assertIn(("unknown", "transient"), trans)
        # Long after the stop both report a stationary object.
        self.assertEqual(base[-1][0], "stationary")
        self.assertEqual(trans[-1][0], "stationary")


if __name__ == "__main__":
    unittest.main()

"""GT path velocity and segments of tools/eval_velocity_gt.py."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "lidar"))
sys.path.insert(0, str(root / "tools"))

try:
    from eval_velocity_gt import (
        PathMotion, central_difference, frame_rows, stats)
except ImportError:  # pragma: no cover - ROS not sourced
    PathMotion = None

FRAME = {"ox": 156.0, "oy": 0.052, "dx": 1, "sy": 1}
S = 1_000_000_000


def item(points=((25.0, -3.0), (8.0, -3.0)), speed=1.2, epoch=10.0):
    return {"motion": {"points": [list(p) for p in points],
                       "speed_mps": speed, "epoch_sim_s": epoch}}


@unittest.skipIf(PathMotion is None, "ROS 2 Python not available")
class PathMotionTest(unittest.TestCase):
    def test_world_position_and_velocity(self):
        p = PathMotion(item(), FRAME)
        np.testing.assert_allclose(p.position(10 * S), [181.0, -2.948])
        np.testing.assert_allclose(p.position(15 * S), [175.0, -2.948])
        np.testing.assert_allclose(p.velocity(15 * S), [-1.2, 0.0])
        # 17 m at 1.2 m/s: stops at 24.17 s, stays at the end point.
        np.testing.assert_allclose(p.position(30 * S), [164.0, -2.948])
        np.testing.assert_allclose(p.velocity(30 * S), [0.0, 0.0])
        np.testing.assert_allclose(p.velocity(9 * S), [0.0, 0.0])
        self.assertEqual(p.phase(9 * S), "before_start")
        self.assertEqual(p.phase(30 * S), "after_stop")
        self.assertAlmostEqual(p.since_stop_s(30 * S), 30 - 10 - 17 / 1.2)

    def test_two_segments(self):
        p = PathMotion(item(points=((0, 0), (3, 0), (3, 4)), speed=1.0,
                            epoch=0.0), {"ox": 0, "oy": 0})
        np.testing.assert_allclose(p.velocity(int(4 * S)), [0.0, 1.0])
        np.testing.assert_allclose(p.position(int(4 * S)), [3.0, 1.0])

    def test_central_difference_real_stamps(self):
        stamps = [0, S // 10, 3 * S // 10]       # a 200 ms gap
        centers = [np.array([0.0, 0]), np.array([0.1, 0]),
                   np.array([0.3, 0])]
        cd = central_difference(stamps, centers)
        self.assertIsNone(cd[0])
        np.testing.assert_allclose(cd[1], [1.0, 0.0])


@unittest.skipIf(PathMotion is None, "ROS 2 Python not available")
class FrameRowsTest(unittest.TestCase):
    def test_segments_and_published(self):
        path = PathMotion(item(points=((0, 0), (10, 0)), speed=1.0,
                               epoch=1.0), {"ox": 0, "oy": 0})
        entries = []
        for k in range(0, 140):            # 0 .. 13.9 s at 10 Hz
            t = k * S // 10
            pos = path.position(t)
            e = {"index": k, "stamp_ns": t, "gt_center": pos.tolist()}
            if k >= 3:
                e.update({"track_id": 4, "raw_velocity": path.velocity(t),
                          "state": "unknown" if k == 50 else "moving",
                          "reason": "shape_changing" if k == 50 else None,
                          "center": pos})
            entries.append(e)
        args = SimpleNamespace(init_s=1.0, start_s=1.0, stop_s=1.5,
                               gap_ms=150.0, gap_hold_s=0.5)
        rows = frame_rows(entries, path, args)
        seg = {r["index"]: r.get("segment") for r in rows}
        self.assertEqual(seg[5], "before_start")
        self.assertEqual(seg[10], "before_start")   # t == start: not moved
        self.assertEqual(seg[11], "transition")     # window holds the start
        self.assertEqual(seg[12], "init")           # track age 0.9 s
        self.assertEqual(seg[18], "after_start")    # 0.8 s after start
        self.assertEqual(seg[40], "steady")
        self.assertEqual(seg[115], "stop_decay")    # stop at 11.0 s
        self.assertEqual(seg[130], "after_stop")
        self.assertIsNone(rows[50]["published"])     # unknown -> null
        self.assertAlmostEqual(rows[40]["err_vector"], 0.0)
        self.assertEqual(stats([0.1, -0.3])["mae"], 0.2)


if __name__ == "__main__":
    unittest.main()

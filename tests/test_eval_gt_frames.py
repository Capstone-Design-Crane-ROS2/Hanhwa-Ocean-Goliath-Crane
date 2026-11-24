"""GT frame selection of tools/eval_objects_gt.py: one GT frame per LiDAR
scan set, boxes at the scan stamp (2nd dataset), 1st dataset unchanged."""

import sys
import unittest
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "lidar"))
sys.path.insert(0, str(root / "tools"))

try:
    from eval_objects_gt import (
        UnsupportedBag, gt_at_lidar, gt_boxes, select_gt_frames,
        select_scan_frames)
except ImportError:  # pragma: no cover - ROS not sourced
    select_scan_frames = None

SCENE = {"eval_frame": {"ox": 156.0, "oy": 0.052, "dx": 1, "sy": 1},
         "ground_truth": []}
CORNERS = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
           [0, 0, 2], [1, 0, 2], [1, 1, 2], [0, 1, 2]]


def frame(scans, dt, x=10.0, x_lidar=None, index=0):
    """A 2nd-dataset style GT frame with one moving object."""
    obj = {"object_id": "P", "object_class": "person",
           "ground_truth": {"x": x, "y": -3.0, "z": 0.0},
           "box_corners_world": CORNERS,
           "box_center_world": [0.5, 0.5, 1.0]}
    if x_lidar is not None:
        obj["ground_truth_at_lidar"] = {"x": x_lidar, "y": -3.0, "z": 0.0,
                                        "stamp_ns": scans[-1]}
    return {"lidar_stamps_ns": list(scans), "lidar_dt_s": dt,
            "measurement_frame_index": index, "objects": [obj]}


@unittest.skipIf(select_scan_frames is None, "ROS 2 Python not available")
class SelectScanFramesTest(unittest.TestCase):
    def test_one_frame_per_scan_prefers_zero_dt(self):
        a = [frame([100], -0.033, index=0), frame([100], 0.0, index=1),
             frame([100], 0.033, index=2), frame([200], 0.033, index=3),
             frame([200], -0.033, index=4)]
        kept, removed = select_scan_frames(a)
        self.assertEqual(removed, 3)
        self.assertEqual([g["measurement_frame_index"] for g in kept],
                         [1, 3])  # dt 0; tie -> first
        self.assertEqual([g["lidar_stamps_ns"] for g in kept],
                         [[100], [200]])

    def test_first_dataset_unchanged(self):
        # Three scans per frame, every scan set once: nothing removed.
        a = [frame([1, 2, 3], -0.01), frame([4, 5, 6], 0.02)]
        kept, removed = select_scan_frames(a)
        self.assertEqual(removed, 0)
        self.assertEqual(kept, a)


@unittest.skipIf(select_scan_frames is None, "ROS 2 Python not available")
class GtAtLidarTest(unittest.TestCase):
    def test_zero_dt_unchanged(self):
        g = frame([100], 0.0, x_lidar=10.0)
        self.assertIs(gt_at_lidar(g, SCENE), g)

    def test_box_moved_to_scan_stamp(self):
        g = frame([100], 0.033, x=10.0, x_lidar=9.96)
        out = gt_at_lidar(g, SCENE)
        moved = np.array(out["objects"][0]["box_corners_world"])
        np.testing.assert_allclose(moved - np.array(CORNERS),
                                   [[-0.04, 0, 0]] * 8, atol=1e-9)
        np.testing.assert_allclose(out["objects"][0]["box_center_world"],
                                   [0.46, 0.5, 1.0])
        # Input not modified; gt_boxes uses the moved box.
        self.assertEqual(g["objects"][0]["box_corners_world"], CORNERS)
        (_, _, corners), = gt_boxes(out, SCENE)
        np.testing.assert_allclose(corners, moved)

    def test_mirrored_eval_frame(self):
        scene = {"eval_frame": {"ox": 0.0, "oy": 0.0, "dx": -1, "sy": -1}}
        g = frame([100], 0.033, x=10.0, x_lidar=9.9)
        g["objects"][0]["ground_truth_at_lidar"]["y"] = -2.8
        shift = gt_at_lidar(g, scene)["objects"][0]["gt_shift_to_lidar"]
        np.testing.assert_allclose(shift, [0.1, -0.2, 0.0], atol=1e-9)

    def test_first_dataset_objects_unchanged(self):
        # No ground_truth_at_lidar / per-frame boxes: nothing to shift.
        g = frame([1, 2, 3], 0.02)
        del g["objects"][0]["box_corners_world"]
        out = gt_at_lidar(g, SCENE)
        self.assertEqual(out["objects"], g["objects"])


@unittest.skipIf(select_scan_frames is None, "ROS 2 Python not available")
class GtFrameModeTest(unittest.TestCase):
    def frames(self):
        return [frame([100], -0.033, x=10.0, x_lidar=10.04, index=0),
                frame([100], 0.0, x=10.0, x_lidar=10.0, index=1),
                frame([100], 0.033, x=10.0, x_lidar=9.96, index=2),
                frame([200], 0.033, x=9.0, x_lidar=8.96, index=3)]

    def test_camera_mode_is_the_previous_behavior(self):
        gts = self.frames()
        out, selection = select_gt_frames(gts, SCENE)
        self.assertIs(out, gts)            # default: unchanged, all frames
        self.assertIsNone(selection)
        out, _ = select_gt_frames(gts, SCENE, "camera")
        self.assertIs(out, gts)

    def test_scan_mode(self):
        out, selection = select_gt_frames(self.frames(), SCENE, "scan")
        self.assertEqual([g["measurement_frame_index"] for g in out], [1, 3])
        self.assertEqual(selection, {"measuring_frames": 4, "scan_frames": 2,
                                     "duplicates_removed": 2,
                                     "objects_shifted_to_lidar": 1})
        shift = out[1]["objects"][0]["gt_shift_to_lidar"]
        np.testing.assert_allclose(shift, [-0.04, 0, 0], atol=1e-9)

    def test_bad_mode(self):
        with self.assertRaises(ValueError):
            select_gt_frames(self.frames(), SCENE, "lidar")

    def test_unsupported_bag_is_a_key_error(self):
        # Callers that caught the previous KeyError('/tf_static') still do.
        exc = UnsupportedBag("bag: no /tf_static")
        self.assertIsInstance(exc, KeyError)
        self.assertEqual(str(exc), "bag: no /tf_static")


if __name__ == "__main__":
    unittest.main()

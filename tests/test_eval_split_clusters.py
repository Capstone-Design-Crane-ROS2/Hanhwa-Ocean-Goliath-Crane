"""Split / merge metrics of tools/eval_split_clusters.py (pure functions)."""

import sys
import unittest
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "lidar"))
sys.path.insert(0, str(root / "tools"))

try:
    from eval_split_clusters import frame_split, summarize_gt
except ImportError:  # pragma: no cover - ROS not sourced
    frame_split = None


def box(x0, x1, y0, y1, z0=0.0, z1=2.0):
    """Axis-aligned corners, bottom 4 then top 4 (eval_objects_gt order)."""
    bottom = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]]
    return np.array(bottom + [[x, y, z1] for x, y, _ in bottom])


def blob(center, n=20, spread=0.1, seed=0):
    rng = np.random.default_rng(seed)
    return np.asarray(center) + rng.uniform(-spread, spread, (n, 3))


@unittest.skipIf(frame_split is None, "ROS 2 Python not available")
class FrameSplitTest(unittest.TestCase):
    def test_body_and_fragment(self):
        truck = box(0, 8, 0, 2.5, 0, 3)
        body = blob([3, 1.2, 1.0], n=200, spread=1.0)
        roof = blob([6, 1.2, 2.8], n=20)
        per_gt, merged = frame_split([body, roof], [("T", "truck", truck)],
                                     {"T": 0})
        rec = per_gt["T"]
        self.assertEqual(rec["candidates"], [0, 1])
        self.assertEqual(len(rec["fragments"]), 1)
        frag = rec["fragments"][0]
        self.assertEqual(frag["object"], 1)
        self.assertAlmostEqual(frag["purity"], 1.0)
        self.assertAlmostEqual(frag["local_center"][0], 6, delta=0.15)
        self.assertEqual(merged, [])

    def test_structure_through_box_is_not_fragment(self):
        # A long wall: 10 of its 200 points cross the GT box.
        person = box(0, 0.4, 0, 0.5)
        wall = np.column_stack([np.linspace(-10, 10, 200),
                                np.full(200, 0.3), np.full(200, 1.0)])
        body = blob([0.2, 0.25, 1.0], n=30, spread=0.1)
        per_gt, _ = frame_split([body, wall], [("P", "person", person)],
                                {"P": 0})
        self.assertEqual(per_gt["P"]["fragments"], [])
        self.assertEqual(per_gt["P"]["candidates"], [0])

    def test_small_piece_below_min_points(self):
        truck = box(0, 8, 0, 2.5, 0, 3)
        per_gt, _ = frame_split(
            [blob([3, 1, 1], n=100, spread=1.0), blob([6, 1, 2.8], n=3)],
            [("T", "truck", truck)], {"T": 0}, min_points=5)
        self.assertEqual(per_gt["T"]["fragments"], [])

    def test_merged_two_gt(self):
        a, b = box(0, 0.4, 0, 0.5), box(0.6, 1.0, 0, 0.5)
        both = np.vstack([blob([0.2, 0.25, 1], n=30, spread=0.1),
                          blob([0.8, 0.25, 1], n=30, spread=0.1, seed=1)])
        per_gt, merged = frame_split([both], [("A", "person", a),
                                              ("B", "person", b)],
                                     {"A": 0})
        self.assertEqual(len(merged), 1)
        self.assertEqual(sorted(merged[0]["gt_ids"]), ["A", "B"])
        # Not a fragment of A: the merged cluster is mostly outside A.
        self.assertEqual(per_gt["A"]["fragments"], [])


@unittest.skipIf(frame_split is None, "ROS 2 Python not available")
class SummarizeTest(unittest.TestCase):
    def frames(self, pattern):
        """pattern[k]: (fragment track id or None or 'none', body track)."""
        out = []
        for k, (frag_tid, body_tid) in enumerate(pattern):
            frags = [] if frag_tid == "none" else [{
                "object": 1, "points": 20, "in_box": 20, "purity": 1.0,
                "share_of_box": 0.2, "local_center": [6.1, 1.2, 2.8],
                "size": [3, 1, 0.2], "bottom_z": 2.7, "track_id": frag_tid}]
            out.append({"index": k, "stamp_ns": k * 100_000_000,
                        "class": "truck", "observable": True, "body": 0,
                        "candidates": [0, 1] if frags else [0],
                        "fragments": frags, "clustered_in_box": 100,
                        "body_track_id": body_tid})
        return out

    def test_persistent_fragment_with_track(self):
        s = summarize_gt(self.frames([(None, 5), (None, 5), (7, 5), (7, 5),
                                      ("none", 5)]))
        self.assertEqual(s["split_frames"], 4)
        self.assertAlmostEqual(s["split_frame_ratio"], 0.8)
        self.assertEqual(s["clusters_per_frame"], {"0": 0, "1": 1, "2": 4,
                                                   "3+": 0})
        self.assertEqual(s["persistent_fragment_cells"], 1)
        self.assertEqual(s["fragment_track_ids"], [7])
        self.assertEqual(s["frames_with_confirmed_fragment_track"], 2)
        self.assertEqual(s["body_fragment_swaps"], [])

    def test_body_fragment_swap(self):
        # The GT match moves to the piece that carried track 7.
        s = summarize_gt(self.frames([(7, 5), (7, 5), (5, 7), (5, 7)]))
        # One change (frame 2), not one per frame with the old ID.
        self.assertEqual([w["index"] for w in s["body_fragment_swaps"]],
                         [2])

    def test_transient_fragment(self):
        s = summarize_gt(self.frames([("none", 5)] * 9 + [(None, 5)]))
        self.assertEqual(s["persistent_fragment_cells"], 0)
        self.assertEqual(s["transient_fragment_cells"], 1)


if __name__ == "__main__":
    unittest.main()

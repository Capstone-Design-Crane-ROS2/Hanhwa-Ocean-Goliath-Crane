"""Unit tests for the tracking metrics of tools/eval_tracking_gt.py."""

import sys
import unittest
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "lidar"))
sys.path.insert(0, str(root / "tools"))

from eval_tracking_gt import object_events, summarize_object  # noqa: E402


def entries(track_ids, detected=None, speed=0.0):
    """Per-frame entries of one GT object at 10 Hz."""
    out = []
    for k, tid in enumerate(track_ids):
        entry = {
            "index": k,
            "stamp_ns": k * 100_000_000,
            "class": "person",
            "status": "ok",
            "detected": detected[k] if detected else True,
            "track_id": tid,
            "gt_center": [speed * k * 0.1, 0.0, 0.0],
        }
        if tid is not None:
            entry["track_velocity"] = [speed, 0.0, 0.0]
        out.append(entry)
    return out


class SummarizeObjectTest(unittest.TestCase):
    def test_stable(self):
        result = summarize_object(entries([None, None, 3, 3, 3, 3]))
        self.assertEqual(result["id_switches"], 0)
        self.assertEqual(result["fragments"], 0)
        self.assertEqual(result["tracked"], 4)
        self.assertEqual(result["init_delay_frames"], 2)
        self.assertAlmostEqual(result["main_id_share"], 4 / 6)

    def test_switch_and_fragment(self):
        # 3 -> gap -> 3 is a fragment only; 3 -> 5 is a switch.
        result = summarize_object(entries([3, 3, None, 3, 5, 5]))
        self.assertEqual(result["id_switches"], 1)
        self.assertEqual(result["fragments"], 1)
        self.assertEqual(result["track_ids"], [3, 5])

    def test_switch_across_gap(self):
        result = summarize_object(entries([3, None, None, 4]))
        self.assertEqual(result["id_switches"], 1)
        self.assertEqual(result["fragments"], 1)

    def test_undetected_frames(self):
        result = summarize_object(
            entries([None, None, 2, 2], detected=[False, True, True, True])
        )
        self.assertEqual(result["detected"], 3)
        self.assertEqual(result["init_delay_frames"], 1)

    def test_velocity_error_only_for_moving(self):
        moving = summarize_object(entries([1] * 5, speed=2.0))
        self.assertEqual(len(moving["velocity_errors"]), 3)
        self.assertAlmostEqual(max(moving["velocity_errors"]), 0.0)
        still = summarize_object(entries([1] * 5, speed=0.0))
        self.assertEqual(still["velocity_errors"], [])


class ObjectEventsTest(unittest.TestCase):
    def test_events(self):
        e = entries([None, 3, 3, None, 3, 5, 5],
                    detected=[True, True, True, False, True, True, True])
        e[3]["status"] = "split"
        events = object_events(e)
        self.assertEqual([x["type"] for x in events],
                         ["confirmed", "lost", "reacquired", "switch"])
        lost = events[1]
        self.assertEqual((lost["index"], lost["status"], lost["detected"]),
                         (3, "split", False))
        switch = events[3]
        self.assertEqual((switch["from"], switch["to"], switch["index"],
                          switch["stamp_ns"]), (3, 5, 5, 500_000_000))
        self.assertEqual(switch["window"],
                         [(3, "split", None), (4, "ok", 3), (5, "ok", 5),
                          (6, "ok", 5)])

    def test_gap_frames(self):
        e = entries([3, 4])
        e[1]["index"] = 4  # frames 1-3 had no points (not in entries)
        events = object_events(e)
        self.assertEqual(events[1]["type"], "switch")
        self.assertEqual(events[1]["gap_frames"], 3)

if __name__ == "__main__":
    unittest.main()

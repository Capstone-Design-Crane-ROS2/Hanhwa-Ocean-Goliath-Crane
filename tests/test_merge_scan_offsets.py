"""lidar_merge_360 scan selection: default 3-scan merge, scan_offsets:=[0]
and camera_trigger:=false (every single full scan published).

The node is driven directly (callbacks with hand-made messages, publishers
replaced by lists), no DDS traffic.
"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lidar"))

try:
    import rclpy
    from builtin_interfaces.msg import Time
    from sensor_msgs.msg import Image
    from std_msgs.msg import Header
    from lidar_merge_360 import LidarMerger, make_cloud
except ImportError:  # pragma: no cover - ROS not sourced
    rclpy = None

P = 33_333_333  # ns


def stamp(ns):
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


@unittest.skipIf(rclpy is None, "ROS 2 Python packages not available")
class MergeOffsetsTest(unittest.TestCase):
    def make(self, *ros_args):
        # LidarMerger takes no arguments: parameters come from the global
        # --ros-args of rclpy.init.
        rclpy.init(args=["--ros-args", *ros_args])
        self.addCleanup(rclpy.shutdown)
        node = LidarMerger()
        self.addCleanup(node.destroy_node)
        clouds, syncs = [], []
        node.cloud_pub.publish = clouds.append
        node.sync_pub.publish = lambda m: syncs.append(json.loads(m.data))
        return node, clouds, syncs

    def scan(self, node, ns, n=4):
        pts = np.arange(1, 3 * n + 1, dtype=np.float32).reshape(n, 3)
        node.lidar_callback(make_cloud(pts, stamp(ns), "OS1_A"))

    def camera(self, node, ns):
        node.camera_callback(Image(header=Header(stamp=stamp(ns))))

    def test_default_merges_three_scans(self):
        node, clouds, syncs = self.make()
        self.assertEqual(node.offsets, [-1, 0, 1])
        base = 1_000 * P
        self.camera(node, base)
        for k in (-1, 0, 1):
            self.scan(node, base + k * P)
        self.assertEqual(len(clouds), 1)
        self.assertEqual(clouds[0].width, 12)
        self.assertEqual(syncs[0]["lidar_stamps_ns"],
                         [base - P, base, base + P])

    def test_single_full_scan(self):
        node, clouds, syncs = self.make("-p", "scan_offsets:=[0]",
                                        "-p", "log_skips:=false")
        base = 3_000 * P
        # 2nd dataset pattern: scans every 3 P, cameras every P.
        for j in range(3):
            scan_ns = base + 3 * j * P
            self.scan(node, scan_ns)
            for k in range(3):
                self.camera(node, scan_ns + k * P)
        self.scan(node, base + 9 * P)
        stamps = [s["lidar_stamps_ns"] for s in syncs]
        self.assertEqual(stamps, [[base], [base + 3 * P], [base + 6 * P]])
        self.assertEqual([s["camera_stamp_ns"] for s in syncs],
                         [base, base + 3 * P, base + 6 * P])
        self.assertEqual(len(clouds), 3)
        self.assertEqual(node.counts["skip_missing_scan"], 6)
        # Each scan is published once and unchanged in size.
        self.assertTrue(all(c.width == 4 for c in clouds))


    # ---- camera_trigger:=false ------------------------------------------

    def make_lidar_mode(self):
        return self.make("-p", "scan_offsets:=[0]",
                         "-p", "camera_trigger:=false",
                         "-p", "log_skips:=false")

    def test_camera_trigger_false_needs_single_scan(self):
        rclpy.init(args=["--ros-args", "-p", "camera_trigger:=false"])
        self.addCleanup(rclpy.shutdown)
        with self.assertRaises(ValueError):
            LidarMerger()

    def test_every_scan_published_once_without_camera(self):
        node, clouds, syncs = self.make_lidar_mode()
        base = 3_000 * P
        scans = [base + 3 * j * P for j in range(4)]
        # 2nd dataset pattern, camera frame of scan 1 missing in the bag;
        # cameras arrive after their scan.
        for j, scan_ns in enumerate(scans):
            self.scan(node, scan_ns)
            for k in range(3):
                if not (j == 1 and k == 0):
                    self.camera(node, scan_ns + k * P)
        stamps = [c.header.stamp.sec * 10**9 + c.header.stamp.nanosec
                  for c in clouds]
        self.assertEqual(stamps, scans)
        self.assertTrue(all(c.header.frame_id == "OS1_A" for c in clouds))
        self.assertTrue(all(c.width == 4 for c in clouds))
        self.assertEqual([s["camera_stamp_ns"] for s in syncs],
                         [scans[0], scans[2], scans[3]])
        self.assertEqual([s["lidar_stamps_ns"] for s in syncs],
                         [[scans[0]], [scans[2]], [scans[3]]])
        self.assertTrue(all(s["point_count"] == 4 for s in syncs))
        node.report()
        self.assertEqual(node.counts["published"], 4)
        self.assertEqual(node.counts["sync_published"], 3)
        self.assertEqual(node.counts["scan_no_camera"], 1)
        # 2 non-scan cameras per scan, minus the last scan's two (pending).
        self.assertEqual(node.counts["camera_no_scan"], 6)

    def test_camera_before_scan_is_paired(self):
        node, clouds, syncs = self.make_lidar_mode()
        base = 3_000 * P
        self.camera(node, base + 1_000_000)  # 1 ms off, within tolerance
        self.assertEqual(syncs, [])
        self.scan(node, base)
        self.assertEqual(len(clouds), 1)
        self.assertEqual(syncs[0]["camera_stamp_ns"], base + 1_000_000)
        self.assertEqual(syncs[0]["lidar_stamps_ns"], [base])

    def test_one_sync_per_scan_and_no_duplicates(self):
        node, clouds, syncs = self.make_lidar_mode()
        base = 3_000 * P
        self.scan(node, base)
        self.scan(node, base)                    # duplicate scan
        self.camera(node, base)
        self.camera(node, base)                  # duplicate camera
        self.camera(node, base + 1_000_000)      # second camera, same scan
        self.assertEqual(len(clouds), 1)
        self.assertEqual(len(syncs), 1)
        self.assertEqual(node.counts["dup_lidar"], 1)
        self.assertEqual(node.counts["dup_camera"], 1)
        self.scan(node, base + 3 * P)
        self.assertEqual(node.counts["camera_no_scan"], 1)

    def test_backwards_stamp_resets_and_keeps_publishing(self):
        node, clouds, syncs = self.make_lidar_mode()
        base = 3_000 * P
        self.scan(node, base)
        self.scan(node, base - 30 * P)           # bag restart
        self.camera(node, base - 30 * P)
        self.assertEqual(len(clouds), 2)
        self.assertEqual(node.counts["reset"], 1)
        self.assertEqual(node.counts["scan_no_camera"], 1)
        self.assertEqual(syncs[-1]["lidar_stamps_ns"], [base - 30 * P])

if __name__ == "__main__":
    unittest.main()

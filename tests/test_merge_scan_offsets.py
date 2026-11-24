"""lidar_merge_360 scan selection: default 3-scan merge and scan_offsets:=[0].

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


if __name__ == "__main__":
    unittest.main()

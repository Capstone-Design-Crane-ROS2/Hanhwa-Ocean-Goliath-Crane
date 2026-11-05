#!/usr/bin/env python3
"""Integration monitor: checks the object node's outputs while a bag plays.

Records the stamps of the input cloud and of every output, then writes a JSON
report: frame counts, missing and duplicate stamps per topic, frame IDs, and
the object lists needed for a position/size review.

Usage:
    python3 tools/object_topic_monitor.py --out report.json
Stop with Ctrl-C or SIGTERM; the report is written on exit.
"""

import argparse
import json
from collections import Counter

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


def stamp_ns(header):
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


class ObjectTopicMonitor(Node):
    def __init__(self, input_topic):
        super().__init__("object_topic_monitor")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.stamps = {
            name: [] for name in ("input", "objects", "markers", "points")
        }
        self.frame_ids = {name: Counter() for name in self.stamps}
        self.frames = []
        self.marker_counts = []
        self.points_by_stamp = {}
        self.merge_sync = {}

        self.create_subscription(
            PointCloud2, input_topic, self.on_input, qos
        )
        self.create_subscription(
            String, "/lidar/objects", self.on_objects, qos
        )
        self.create_subscription(
            MarkerArray, "/lidar/objects/markers", self.on_markers, qos
        )
        self.create_subscription(
            PointCloud2, "/lidar/objects/points", self.on_points, qos
        )
        self.create_subscription(
            String, "/lidar/merge_sync", self.on_merge_sync, qos
        )

    def _note(self, name, stamp, frame_id):
        self.stamps[name].append(stamp)
        self.frame_ids[name][frame_id] += 1

    def on_input(self, msg):
        self._note("input", stamp_ns(msg.header), msg.header.frame_id)

    def on_objects(self, msg):
        data = json.loads(msg.data)
        self._note("objects", data["stamp_ns"], data["frame_id"])
        self.frames.append(data)

    def on_markers(self, msg):
        first = msg.markers[0]
        self._note("markers", stamp_ns(first.header), first.header.frame_id)
        boxes = sum(m.type == Marker.LINE_LIST for m in msg.markers)
        self.marker_counts.append((stamp_ns(first.header), boxes))

    def on_points(self, msg):
        self._note("points", stamp_ns(msg.header), msg.header.frame_id)
        self.points_by_stamp[stamp_ns(msg.header)] = msg.width

    def on_merge_sync(self, msg):
        data = json.loads(msg.data)
        self.merge_sync[data["lidar_stamps_ns"][-1]] = data["camera_stamp_ns"]

    def report(self):
        reference = set(self.stamps["input"])
        topics = {}
        for name, stamps in self.stamps.items():
            unique = set(stamps)
            topics[name] = {
                "messages": len(stamps),
                "unique_stamps": len(unique),
                "duplicates": len(stamps) - len(unique),
                "missing_vs_input": len(reference - unique),
                "extra_vs_input": len(unique - reference),
                "in_order": stamps == sorted(stamps),
                "frame_ids": dict(self.frame_ids[name]),
            }

        # Cross-checks between the outputs of one frame.
        boxes = dict(self.marker_counts)
        box_mismatch = sum(
            boxes.get(f["stamp_ns"]) != f["object_count"] for f in self.frames
        )
        points_mismatch = sum(
            self.points_by_stamp.get(f["stamp_ns"])
            != sum(o["point_count"] for o in f["objects"])
            for f in self.frames
        )
        counts = [f["object_count"] for f in self.frames]
        return {
            "topics": topics,
            "marker_box_count_mismatch": box_mismatch,
            "points_count_mismatch": points_mismatch,
            "gravity_aligned_frames": sum(
                f["gravity_aligned"] for f in self.frames
            ),
            "objects_per_frame": {
                "mean": float(np.mean(counts)) if counts else 0.0,
                "min": min(counts, default=0),
                "max": max(counts, default=0),
            },
            "merge_sync_messages": len(self.merge_sync),
            "stamps_with_camera_stamp": len(
                reference & set(self.merge_sync)
            ),
            "frames": self.frames,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--input", default="/lidar/points_preprocessed")
    args = parser.parse_args()

    rclpy.init()
    node = ObjectTopicMonitor(args.input)
    # rclpy's own handlers turn SIGINT/SIGTERM into a context shutdown, which
    # also wakes a spin that is waiting with no messages arriving.
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        # A signal can invalidate the context while spin creates its wait
        # set. That is a normal shutdown; anything else is a real error.
        if rclpy.ok():
            raise
    finally:
        with open(args.out, "w", encoding="utf8") as handle:
            json.dump(node.report(), handle)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

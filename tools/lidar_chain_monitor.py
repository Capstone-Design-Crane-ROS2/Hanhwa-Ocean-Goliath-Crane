#!/usr/bin/env python3
"""Frame delivery check of the LiDAR chain while a bag plays.

Records the header stamp and receive time of every message on

    /point_cloud_A                raw scans (the bag)
    /point_cloud_360              merge output
    /lidar/points_roi             ROI output
    /lidar/points_voxel           voxel output
    /lidar/points_preprocessed    ground output (launch remap)
    /lidar/objects                object node JSON (stamp_ns)
    /lidar/merge_sync             camera <-> scan pairs

and writes a JSON report per topic: messages, unique stamps, duplicates,
stamps missing / extra against a reference topic, order, and the receive
latency from the reference message to that topic's message of the same
stamp. The reference is /point_cloud_A for one-scan data
(camera_trigger:=false) and /point_cloud_360 otherwise (--reference).

Only subscribes; changes nothing. The raw-scan subscription deserializes
every full scan, a small extra load next to the chain.

Usage:
    python3 tools/lidar_chain_monitor.py --out report.json \
        [--reference /point_cloud_A]
Stop with Ctrl-C or SIGTERM; the report is written on exit.
"""

import argparse
import json
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

CLOUD_TOPICS = ("/point_cloud_A", "/point_cloud_360", "/lidar/points_roi",
                "/lidar/points_voxel", "/lidar/points_preprocessed")


def stamp_ns(header):
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


class LidarChainMonitor(Node):
    def __init__(self):
        super().__init__("lidar_chain_monitor")
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=100,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.records = {t: [] for t in CLOUD_TOPICS + ("/lidar/objects",)}
        self.sync = []
        for topic in CLOUD_TOPICS:
            self.create_subscription(
                PointCloud2, topic,
                lambda m, t=topic: self.records[t].append(
                    (stamp_ns(m.header), time.monotonic(), m.width)),
                qos)
        self.create_subscription(String, "/lidar/objects", self.on_objects,
                                 qos)
        self.create_subscription(String, "/lidar/merge_sync", self.on_sync,
                                 qos)

    def on_objects(self, msg):
        data = json.loads(msg.data)
        self.records["/lidar/objects"].append(
            (int(data["stamp_ns"]), time.monotonic(),
             len(data.get("objects", []))))

    def on_sync(self, msg):
        data = json.loads(msg.data)
        self.sync.append((int(data["camera_stamp_ns"]),
                          [int(s) for s in data["lidar_stamps_ns"]]))

    def report(self, reference):
        ref = self.records[reference]
        ref_stamps = {s for s, _, _ in ref}
        ref_time = {}
        for s, t, _ in ref:
            ref_time.setdefault(s, t)
        topics = {}
        for topic, rec in self.records.items():
            stamps = [s for s, _, _ in rec]
            unique = set(stamps)
            lat = [(t - ref_time[s]) * 1e3 for s, t, _ in rec
                   if s in ref_time]

            def pct(q):
                return round(float(np.percentile(lat, q)), 1) if lat else None
            topics[topic] = {
                "messages": len(stamps),
                "unique_stamps": len(unique),
                "duplicates": len(stamps) - len(unique),
                "missing_vs_reference": len(ref_stamps - unique),
                "extra_vs_reference": len(unique - ref_stamps),
                "in_order": stamps == sorted(stamps),
                "latency_from_reference_ms": {
                    "p50": pct(50), "p95": pct(95),
                    "max": round(max(lat), 1) if lat else None},
                "width_or_objects_mean": (
                    round(float(np.mean([w for _, _, w in rec])), 1)
                    if rec else None),
            }
        sync_scans = [ls[-1] for _, ls in self.sync]
        return {
            "reference": reference,
            "topics": topics,
            "merge_sync": {
                "messages": len(self.sync),
                "camera_duplicates": len(self.sync)
                - len({c for c, _ in self.sync}),
                "scan_duplicates": len(sync_scans) - len(set(sync_scans)),
                "max_camera_scan_dt_ms": max(
                    (abs(c - ls[-1]) / 1e6 for c, ls in self.sync),
                    default=None),
                "scans_not_in_reference": len(set(sync_scans) - ref_stamps),
            },
            "objects_stamps": [s for s, _, _ in
                               self.records["/lidar/objects"]],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference", default="/point_cloud_A",
                        choices=CLOUD_TOPICS)
    args = parser.parse_args()

    rclpy.init()
    node = LidarChainMonitor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        with open(args.out, "w", encoding="utf8") as handle:
            json.dump(node.report(args.reference), handle, indent=1)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

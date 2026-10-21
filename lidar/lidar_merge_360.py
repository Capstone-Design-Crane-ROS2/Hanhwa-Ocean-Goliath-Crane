#!/usr/bin/env python3
"""Camera-triggered 360° LiDAR merge.

For each /rgb frame at time t, merges the three 120° scans at
t-P, t, t+P (P = scan period) from /point_cloud_A.

Output contract (consumed by lidar_roi_node.py):
  /point_cloud_360   PointCloud2, xyz float32, frame OS1_A,
                     header.stamp = stamp of the last (t+P) scan
  /lidar/merge_sync  String JSON: camera stamp, 3 LiDAR stamps,
                     point count

A camera frame whose scans are missing is skipped; no other scan is
substituted. Merging runs synchronously in the subscription callback
(~10-15 ms per frame), so there is no work queue to overflow.
"""

import array
import json
import time
from collections import Counter, OrderedDict, deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header, String


XYZ_FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
]


def stamp_to_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def extract_xyz(msg):
    """Read valid XYZ from PointCloud2, respecting field offsets."""
    fields = {field.name: field for field in msg.fields}

    for name in ("x", "y", "z"):
        if name not in fields:
            raise ValueError(f"Missing PointCloud2 field: {name}")
        if fields[name].datatype != PointField.FLOAT32:
            raise ValueError(f"Expected FLOAT32 field: {name}")

    endian = ">" if msg.is_bigendian else "<"

    dtype = np.dtype({
        "names": ["x", "y", "z"],
        "formats": [endian + "f4"] * 3,
        "offsets": [fields[name].offset for name in ("x", "y", "z")],
        "itemsize": msg.point_step,
    })

    cloud = np.ndarray(
        shape=(msg.height, msg.width),
        dtype=dtype,
        buffer=msg.data,
        strides=(msg.row_step, msg.point_step),
    )

    points = np.empty((cloud.size, 3), dtype="<f4")
    points[:, 0] = cloud["x"].reshape(-1)
    points[:, 1] = cloud["y"].reshape(-1)
    points[:, 2] = cloud["z"].reshape(-1)

    valid = np.isfinite(points).all(axis=1)
    valid &= np.any(points != 0.0, axis=1)

    return points[valid]


def make_cloud(points, stamp, frame_id):
    """Build xyz float32 PointCloud2.

    msg.data is assigned an array.array('B'). Assigning bytes makes the
    rclpy-generated setter validate every byte in Python (~260 ms for
    360k points), which saturated the old work queue at 1x playback.
    """
    points = np.ascontiguousarray(points, dtype="<f4")

    msg = PointCloud2()
    msg.header = Header(stamp=stamp, frame_id=frame_id)

    msg.height = 1
    msg.width = len(points)
    msg.fields = XYZ_FIELDS
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.width * msg.point_step
    msg.is_dense = True

    data = array.array("B")
    data.frombytes(points.tobytes())
    msg.data = data

    return msg


class ScanEntry:
    """Received scan with lazily cached XYZ.

    Adjacent camera frames share one scan, so each scan is converted
    at most once, and scans never used are never converted.
    """

    __slots__ = ("msg", "xyz")

    def __init__(self, msg):
        self.msg = msg
        self.xyz = None

    def points(self):
        if self.xyz is None:
            self.xyz = extract_xyz(self.msg)
        return self.xyz


class LidarMerger(Node):

    def __init__(self):
        super().__init__("lidar_merge_360")

        self.declare_parameter("scan_period_ms", 33.333333)
        self.declare_parameter("tolerance_ms", 2.0)
        self.declare_parameter("max_camera_wait_ms", 500.0)

        self.period_ns = round(
            self.get_parameter("scan_period_ms").value * 1e6
        )
        self.tolerance_ns = round(
            self.get_parameter("tolerance_ms").value * 1e6
        )
        # Sensor-time bound on how long a camera frame may wait for its
        # scans. Also sets how much scan history is kept.
        self.max_wait_ns = round(
            self.get_parameter("max_camera_wait_ms").value * 1e6
        )

        qos = QoSProfile(
            depth=50,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.cloud_pub = self.create_publisher(
            PointCloud2, "/point_cloud_360", qos
        )
        self.sync_pub = self.create_publisher(
            String, "/lidar/merge_sync", qos
        )

        self.lidar_sub = self.create_subscription(
            PointCloud2,
            "/point_cloud_A",
            self.lidar_callback,
            qos,
        )
        self.camera_sub = self.create_subscription(
            Image,
            "/rgb",
            self.camera_callback,
            qos,
        )

        # Timestamp -> ScanEntry; insertion order is timestamp order
        # because backwards timestamps reset the buffers.
        self.scans = OrderedDict()
        self.pending_cameras = deque()

        self.last_lidar_stamp = None
        self.last_camera_stamp = None
        self.frame_id = None

        self.counts = Counter()
        self.merge_ms = []
        self.callback_ms = []

        self.get_logger().info(
            "Camera-synchronized LiDAR merger started"
        )

    def reset_buffers(self, reason):
        self.get_logger().warn(
            f"{reason}: clearing synchronization buffers"
        )
        self.counts["reset"] += 1
        self.counts["skip_reset"] += len(self.pending_cameras)

        self.scans.clear()
        self.pending_cameras.clear()
        self.last_lidar_stamp = None
        self.last_camera_stamp = None
        self.frame_id = None

    def lidar_callback(self, msg):
        start = time.perf_counter()
        stamp = stamp_to_ns(msg.header.stamp)

        if self.last_lidar_stamp is not None:
            if stamp == self.last_lidar_stamp:
                self.counts["dup_lidar"] += 1
                return
            if stamp < self.last_lidar_stamp:
                self.reset_buffers("LiDAR timestamp went backwards")

        if self.frame_id is not None and (
            msg.header.frame_id != self.frame_id
        ):
            self.reset_buffers("LiDAR frame_id changed")

        self.frame_id = msg.header.frame_id
        self.last_lidar_stamp = stamp
        self.scans[stamp] = ScanEntry(msg)
        self.counts["lidar_in"] += 1

        self.process()
        self.callback_ms.append((time.perf_counter() - start) * 1e3)

    def camera_callback(self, msg):
        start = time.perf_counter()
        stamp = stamp_to_ns(msg.header.stamp)

        if self.last_camera_stamp is not None:
            if stamp == self.last_camera_stamp:
                self.counts["dup_camera"] += 1
                return
            if stamp < self.last_camera_stamp:
                self.reset_buffers("Camera timestamp went backwards")

        self.last_camera_stamp = stamp
        self.pending_cameras.append(stamp)
        self.counts["camera_in"] += 1

        self.process()
        self.callback_ms.append((time.perf_counter() - start) * 1e3)

    def find_scan(self, target):
        for stamp in self.scans:
            if abs(stamp - target) <= self.tolerance_ns:
                return stamp
        return None

    def process(self):
        """Merge or skip pending camera frames in arrival order."""
        latest = self.last_lidar_stamp

        while self.pending_cameras:
            camera_stamp = self.pending_cameras[0]

            selected = [
                self.find_scan(camera_stamp + k * self.period_ns)
                for k in (-1, 0, 1)
            ]

            if None not in selected:
                self.pending_cameras.popleft()
                self.merge(camera_stamp, selected)
                continue

            if latest is None:
                break

            if latest > camera_stamp + self.max_wait_ns:
                # Camera arrived too late; its scans may be evicted.
                reason = "skip_timeout"
            elif latest > (
                camera_stamp + self.period_ns + self.tolerance_ns
            ):
                # A later scan already arrived, so a required scan is
                # absent. Do not substitute a neighboring scan.
                reason = "skip_missing_scan"
            else:
                break  # wait for later scans

            self.pending_cameras.popleft()
            self.counts[reason] += 1
            self.get_logger().warn(
                f"{reason}: camera {camera_stamp}, "
                f"scans {selected}"
            )

        self.evict()

    def evict(self):
        if self.last_lidar_stamp is None:
            return

        oldest_needed = (
            self.last_lidar_stamp
            - self.max_wait_ns
            - self.period_ns
            - self.tolerance_ns
        )
        if self.pending_cameras:
            oldest_needed = min(
                oldest_needed,
                self.pending_cameras[0]
                - self.period_ns
                - self.tolerance_ns,
            )

        while self.scans:
            stamp = next(iter(self.scans))
            if stamp >= oldest_needed:
                break
            del self.scans[stamp]

    def merge(self, camera_stamp, lidar_stamps):
        start = time.perf_counter()
        entries = [self.scans[stamp] for stamp in lidar_stamps]

        try:
            merged_points = np.concatenate(
                [entry.points() for entry in entries], axis=0
            )
            last = entries[-1].msg.header

            output = make_cloud(
                merged_points, last.stamp, last.frame_id
            )
            self.cloud_pub.publish(output)

        except Exception as exc:
            self.counts["failed"] += 1
            self.get_logger().error(
                f"LiDAR merge failed for camera {camera_stamp}: {exc}"
            )
            return

        metadata = String()
        metadata.data = json.dumps({
            "camera_stamp_ns": camera_stamp,
            "lidar_stamps_ns": lidar_stamps,
            "point_count": len(merged_points),
        })
        self.sync_pub.publish(metadata)

        self.counts["published"] += 1

        elapsed_ms = (time.perf_counter() - start) * 1e3
        self.merge_ms.append(elapsed_ms)
        if elapsed_ms * 1e6 > self.period_ns:
            self.get_logger().warn(
                f"Slow merge: {elapsed_ms:.1f} ms",
                throttle_duration_sec=5.0,
            )

    def report(self):
        def summary(values):
            if not values:
                return "n/a"
            v = np.array(values)
            return (
                f"p50={np.median(v):.1f} "
                f"p95={np.percentile(v, 95):.1f} "
                f"max={v.max():.1f} ms"
            )

        self.get_logger().info(
            f"Counts={dict(self.counts)}, "
            f"pending={len(self.pending_cameras)}, "
            f"merge {summary(self.merge_ms)}, "
            f"callback {summary(self.callback_ms)}"
        )


def main():
    rclpy.init()
    node = LidarMerger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

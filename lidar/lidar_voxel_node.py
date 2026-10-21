#!/usr/bin/env python3
"""ROS2 centroid voxel downsampling node.

Input:  /lidar/points_roi
Output: /lidar/points_voxel
"""

import array
import time

import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
)
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2

from voxel_filter import voxel_downsample


XYZ_FIELDS = [
    PointField(
        name="x", offset=0,
        datatype=PointField.FLOAT32, count=1,
    ),
    PointField(
        name="y", offset=4,
        datatype=PointField.FLOAT32, count=1,
    ),
    PointField(
        name="z", offset=8,
        datatype=PointField.FLOAT32, count=1,
    ),
]


def cloud_to_xyz(msg):
    """Convert PointCloud2 into a (N, 3) float32 array."""
    if msg.width * msg.height == 0:
        return np.empty((0, 3), dtype=np.float32)

    points = point_cloud2.read_points(
        msg,
        field_names=("x", "y", "z"),
        skip_nans=False,
    )

    if isinstance(points, np.ndarray) and points.dtype.names:
        result = np.empty((len(points), 3), dtype=np.float32)
        result[:, 0] = points["x"].reshape(-1)
        result[:, 1] = points["y"].reshape(-1)
        result[:, 2] = points["z"].reshape(-1)
        return result

    return np.asarray(
        list(points), dtype=np.float32
    ).reshape(-1, 3)


def xyz_to_cloud(header, points):
    """Build xyz float32 PointCloud2 without slow bytes setter."""
    pts = np.ascontiguousarray(
        points, dtype="<f4"
    ).reshape(-1, 3)

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = len(pts)
    msg.fields = XYZ_FIELDS
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = len(pts) * 12
    msg.is_dense = True

    data = array.array("B")
    data.frombytes(pts.tobytes())
    msg.data = data

    return msg


class LidarVoxelNode(Node):
    def __init__(self):
        super().__init__("lidar_voxel_node")

        self.declare_parameter(
            "input_topic", "/lidar/points_roi"
        )
        self.declare_parameter(
            "output_topic", "/lidar/points_voxel"
        )
        self.declare_parameter("voxel_size", 0.2)

        input_topic = self.get_parameter(
            "input_topic"
        ).value
        output_topic = self.get_parameter(
            "output_topic"
        ).value

        self.voxel_size = float(
            self.get_parameter("voxel_size").value
        )

        if (
            not np.isfinite(self.voxel_size)
            or self.voxel_size <= 0
        ):
            raise ValueError(
                "voxel_size must be finite and > 0"
            )

        # RELIABLE: large PointCloud2 samples are lost over DDS with
        # BEST_EFFORT (S2 0.5x: 7/58 received). Upstream merge and ROI
        # publishers are RELIABLE, so this is compatible.
        sub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.subscription = self.create_subscription(
            PointCloud2,
            input_topic,
            self.callback,
            sub_qos,
        )

        pub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.publisher = self.create_publisher(
            PointCloud2,
            output_topic,
            pub_qos,
        )

        self.received = 0
        self.published = 0
        self.failed = 0
        self.total_input_points = 0
        self.total_output_points = 0
        self.processing_ms = []

        self.get_logger().info(
            f"Voxel node started: "
            f"{input_topic} -> {output_topic}, "
            f"voxel_size={self.voxel_size:.3f}m"
        )

    def callback(self, msg):
        start = time.perf_counter()
        self.received += 1

        try:
            points = cloud_to_xyz(msg)
            filtered = voxel_downsample(
                points, self.voxel_size
            )

            output = xyz_to_cloud(
                msg.header, filtered
            )
            self.publisher.publish(output)

            self.published += 1
            self.total_input_points += len(points)
            self.total_output_points += len(filtered)

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000.0
            self.processing_ms.append(elapsed_ms)

            if self.published % 30 == 0:
                reduction = (
                    100.0 * (
                        1.0 - len(filtered) / len(points)
                    )
                    if len(points) > 0 else 0.0
                )

                self.get_logger().info(
                    f"frames={self.published} "
                    f"in={len(points)} "
                    f"out={len(filtered)} "
                    f"reduction={reduction:.1f}% "
                    f"processing={elapsed_ms:.1f}ms"
                )

        except Exception as exc:
            self.failed += 1
            self.get_logger().error(
                f"Voxel processing failed: {exc}"
            )

    def report(self):
        if self.processing_ms:
            p50, p95 = np.percentile(
                self.processing_ms, [50, 95]
            )
            max_ms = max(self.processing_ms)
        else:
            p50 = p95 = max_ms = 0.0

        reduction = (
            100.0 * (
                1.0
                - self.total_output_points
                / self.total_input_points
            )
            if self.total_input_points else 0.0
        )

        self.get_logger().info(
            f"Voxel stats: "
            f"received={self.received}, "
            f"published={self.published}, "
            f"failed={self.failed}, "
            f"points_in={self.total_input_points}, "
            f"points_out={self.total_output_points}, "
            f"reduction={reduction:.1f}%, "
            f"p50={p50:.1f}ms, "
            f"p95={p95:.1f}ms, "
            f"max={max_ms:.1f}ms"
        )


def main():
    rclpy.init()
    node = None

    try:
        node = LidarVoxelNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.report()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

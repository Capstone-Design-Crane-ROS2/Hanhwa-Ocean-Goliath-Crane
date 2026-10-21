#!/usr/bin/env python3
"""ROS2 ground removal node (tracking constrained RANSAC).

Input:  /lidar/points_voxel
Output: /lidar/points_nonground
        /lidar/points_ground (optional, publish_ground:=true, for RViz2)

The prior ground plane is computed from the world -> sensor TF
(/tf_static) and ground_z_world. If the TF is missing or invalid, or no
valid plane is found, the input is published unchanged (passthrough).
"""

import time

import numpy as np
import rclpy
import tf2_ros

from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2

from ground_filter import GroundParams, GroundRemover, prior_from_transform
from lidar_voxel_node import cloud_to_xyz, xyz_to_cloud


PARAM_DEFAULTS = {
    name: getattr(GroundParams, name)
    for name in GroundParams.__dataclass_fields__
}


class LidarGroundNode(Node):
    def __init__(self):
        super().__init__("lidar_ground_node")

        self.declare_parameter("input_topic", "/lidar/points_voxel")
        self.declare_parameter("output_topic", "/lidar/points_nonground")
        self.declare_parameter("ground_topic", "/lidar/points_ground")
        self.declare_parameter("publish_ground", False)
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("ground_z_world", 0.0)
        for name, default in PARAM_DEFAULTS.items():
            self.declare_parameter(name, default)

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        ground_topic = self.get_parameter("ground_topic").value
        self.publish_ground = bool(
            self.get_parameter("publish_ground").value
        )
        self.world_frame = self.get_parameter("world_frame").value
        self.ground_z = float(
            self.get_parameter("ground_z_world").value
        )

        params = GroundParams(**{
            name: type(default)(self.get_parameter(name).value)
            for name, default in PARAM_DEFAULTS.items()
        })
        self.remover = GroundRemover(params)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # RELIABLE: large PointCloud2 samples are lost over DDS with
        # BEST_EFFORT. Upstream voxel publisher is RELIABLE.
        sub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        pub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.subscription = self.create_subscription(
            PointCloud2, input_topic, self.callback, sub_qos,
        )
        self.publisher = self.create_publisher(
            PointCloud2, output_topic, pub_qos,
        )
        self.ground_publisher = (
            self.create_publisher(PointCloud2, ground_topic, pub_qos)
            if self.publish_ground else None
        )

        self.received = 0
        self.published = 0
        self.failed = 0
        self.total_input_points = 0
        self.total_output_points = 0
        self.modes = {"fit": 0, "hold": 0, "passthrough": 0}
        self.processing_ms = []

        self.get_logger().info(
            f"Ground node started: {input_topic} -> {output_topic}, "
            f"distance_threshold={params.distance_threshold:.3f}m, "
            f"prior TF {self.world_frame} -> <cloud frame>, "
            f"publish_ground={self.publish_ground}"
        )

    def lookup_prior(self, frame_id):
        """Prior plane from TF, or None if unavailable or invalid."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, frame_id, Time(),
            ).transform
            return prior_from_transform(
                (tf.rotation.x, tf.rotation.y,
                 tf.rotation.z, tf.rotation.w),
                (tf.translation.x, tf.translation.y,
                 tf.translation.z),
                self.ground_z,
            )
        except (tf2_ros.TransformException, ValueError) as exc:
            self.get_logger().warn(
                f"No valid TF {self.world_frame} -> {frame_id} "
                f"({exc}); passthrough",
                throttle_duration_sec=5.0,
            )
            return None

    def callback(self, msg):
        start = time.perf_counter()
        self.received += 1

        try:
            points = cloud_to_xyz(msg)
        except Exception as exc:
            self.failed += 1
            self.get_logger().error(f"Invalid PointCloud2: {exc}")
            return

        try:
            prior = self.lookup_prior(msg.header.frame_id)
            ground, result = self.remover.process(points, prior)
        except Exception as exc:
            # Do not drop the frame: publish the input unchanged.
            self.failed += 1
            self.get_logger().error(f"Ground removal failed: {exc}")
            ground = np.zeros(len(points), dtype=bool)
            result = None

        kept = points[~ground]

        self.publisher.publish(xyz_to_cloud(msg.header, kept))
        if self.ground_publisher is not None:
            self.ground_publisher.publish(
                xyz_to_cloud(msg.header, points[ground])
            )

        self.published += 1
        self.total_input_points += len(points)
        self.total_output_points += len(kept)
        mode = result.mode if result is not None else "passthrough"
        self.modes[mode] += 1

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.processing_ms.append(elapsed_ms)

        if mode != "fit" and result is not None:
            self.get_logger().warn(
                f"Ground {mode}: {result.reason} "
                f"(candidates={result.candidates}, "
                f"inliers={result.inliers})",
                throttle_duration_sec=5.0,
            )
        elif self.published % 30 == 0:
            self.get_logger().info(
                f"frames={self.published} in={len(points)} "
                f"out={len(kept)} ground={int(ground.sum())} "
                f"normal={np.round(result.normal, 4).tolist()} "
                f"height={result.height:.3f}m "
                f"processing={elapsed_ms:.1f}ms"
            )

    def report(self):
        if self.processing_ms:
            p50, p95 = np.percentile(self.processing_ms, [50, 95])
            max_ms = max(self.processing_ms)
        else:
            p50 = p95 = max_ms = 0.0

        removal = (
            100.0 * (
                1.0
                - self.total_output_points / self.total_input_points
            )
            if self.total_input_points else 0.0
        )

        self.get_logger().info(
            f"Ground stats: "
            f"received={self.received}, "
            f"published={self.published}, "
            f"failed={self.failed}, "
            f"fit={self.modes['fit']}, "
            f"hold={self.modes['hold']}, "
            f"passthrough={self.modes['passthrough']}, "
            f"points_in={self.total_input_points}, "
            f"points_out={self.total_output_points}, "
            f"removed={removal:.1f}%, "
            f"p50={p50:.1f}ms, "
            f"p95={p95:.1f}ms, "
            f"max={max_ms:.1f}ms"
        )


def main():
    rclpy.init()
    node = None

    try:
        node = LidarGroundNode()
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

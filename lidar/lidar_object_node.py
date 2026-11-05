#!/usr/bin/env python3
"""ROS2 LiDAR-only object estimation node.

Input:  /lidar/points_preprocessed   PointCloud2
Output: /lidar/objects               String JSON, one per input frame
        /lidar/objects/markers       MarkerArray for RViz2
        /lidar/objects/points        PointCloud2 with an object_id field

All outputs reuse the input header (stamp, frame_id). Every input frame gets
outputs, also when empty or when processing fails, so a consumer can detect
a real drop by a missing stamp.

The world -> cloud TF makes boxes level and enables the height band. If it
is missing, boxes fall back to the cloud frame (gravity_aligned=false), and
the height band is not applied.

Run:
    python3 lidar/lidar_object_node.py
    python3 lidar/lidar_object_node.py --ros-args -p cluster.tolerance:=0.5
"""

import time

import numpy as np
import rclpy
import tf2_ros

from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

from cluster_extractor import ClusterParams, EuclideanClusterExtractor
from lidar_voxel_node import cloud_to_xyz
from object_estimator import (
    EstimatorParams,
    IDENTITY_POSE,
    ObjectFrame,
    ReferencePose,
    estimate_objects,
    height_band_mask,
)
from object_messages import frame_to_json, frame_to_markers, labeled_cloud


# Height band above ground: keeps the tallest GT object (9.19 m) and drops
# roofs and upper walls, about half of the points on the S2/S4/S7 scenes.
DEFAULT_CROP_MIN_HEIGHT = -1.0
DEFAULT_CROP_MAX_HEIGHT = 10.0

STARTUP_ONLY = (
    "input_topic", "output_topic", "marker_topic", "points_topic",
    "world_frame", "ground_z_world",
)


def stamp_to_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class LidarObjectNode(Node):
    def __init__(self):
        super().__init__("lidar_object_node")

        input_topic = self.declare_parameter(
            "input_topic", "/lidar/points_preprocessed"
        ).value
        output_topic = self.declare_parameter(
            "output_topic", "/lidar/objects"
        ).value
        marker_topic = self.declare_parameter(
            "marker_topic", "/lidar/objects/markers"
        ).value
        points_topic = self.declare_parameter(
            "points_topic", "/lidar/objects/points"
        ).value
        self.world_frame = self.declare_parameter(
            "world_frame", "world"
        ).value
        self.ground_z = float(self.declare_parameter(
            "ground_z_world", 0.0
        ).value)

        self.declare_parameter(
            "cluster.tolerance", ClusterParams.tolerance
        )
        self.declare_parameter(
            "cluster.min_points", ClusterParams.min_points
        )
        self.declare_parameter(
            "cluster.max_points", ClusterParams.max_points
        )
        self.declare_parameter(
            "estimate.center_method", EstimatorParams.center_method
        )
        self.declare_parameter(
            "estimate.max_footprint", EstimatorParams.max_footprint
        )
        self.declare_parameter(
            "estimate.max_height", EstimatorParams.max_height
        )
        self.declare_parameter("crop.min_height", DEFAULT_CROP_MIN_HEIGHT)
        self.declare_parameter("crop.max_height", DEFAULT_CROP_MAX_HEIGHT)
        self.declare_parameter("marker.show_labels", True)

        self._apply_settings(self._read_settings())
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # RELIABLE: large PointCloud2 samples are lost over DDS with
        # BEST_EFFORT. Upstream ground publisher is RELIABLE.
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
        self.object_publisher = self.create_publisher(
            String, output_topic, pub_qos,
        )
        self.marker_publisher = self.create_publisher(
            MarkerArray, marker_topic, pub_qos,
        )
        self.points_publisher = self.create_publisher(
            PointCloud2, points_topic, pub_qos,
        )

        self.received = 0
        self.published = 0
        self.failed = 0
        self.no_tf_frames = 0
        self.total_objects = 0
        self.total_rejected = 0
        self.cluster_ms = []
        self.estimate_ms = []
        self.total_ms = []

        self.get_logger().info(
            f"Object node started: {input_topic} -> {output_topic}, "
            f"{marker_topic}, {points_topic}; {self._settings_text()}"
        )

    def _read_settings(self, overrides=None):
        """Validated settings from parameters plus pending overrides."""
        overrides = overrides or {}

        def value(name):
            if name in overrides:
                return overrides[name]
            return self.get_parameter(name).value

        crop_min = float(value("crop.min_height"))
        crop_max = float(value("crop.max_height"))
        if not (np.isfinite(crop_min) and np.isfinite(crop_max)):
            raise ValueError("crop heights must be finite")
        if crop_min >= crop_max:
            raise ValueError("crop.min_height must be < crop.max_height")

        return {
            "cluster": ClusterParams(
                tolerance=float(value("cluster.tolerance")),
                min_points=int(value("cluster.min_points")),
                max_points=int(value("cluster.max_points")),
            ),
            "estimate": EstimatorParams(
                center_method=str(value("estimate.center_method")),
                max_footprint=float(value("estimate.max_footprint")),
                max_height=float(value("estimate.max_height")),
            ),
            "crop": (crop_min, crop_max),
            "show_labels": bool(value("marker.show_labels")),
        }

    def _apply_settings(self, settings):
        self.extractor = EuclideanClusterExtractor(settings["cluster"])
        self.estimator_params = settings["estimate"]
        self.crop = settings["crop"]
        self.show_labels = settings["show_labels"]

    def _settings_text(self):
        cluster = self.extractor.params
        estimate = self.estimator_params
        return (
            f"tolerance={cluster.tolerance}m "
            f"points=[{cluster.min_points}, {cluster.max_points}] "
            f"crop={self.crop}m center={estimate.center_method} "
            f"max_footprint={estimate.max_footprint}m "
            f"max_height={estimate.max_height}m"
        )

    def _on_set_parameters(self, params):
        overrides = {}
        for param in params:
            if param.name in STARTUP_ONLY:
                return SetParametersResult(
                    successful=False,
                    reason=f"{param.name} can only be set at startup",
                )
            overrides[param.name] = param.value
        try:
            settings = self._read_settings(overrides)
        except (TypeError, ValueError) as exc:
            return SetParametersResult(successful=False, reason=str(exc))
        self._apply_settings(settings)
        self.get_logger().info(f"Settings changed: {self._settings_text()}")
        return SetParametersResult(successful=True)

    def lookup_pose(self, frame_id):
        """ReferencePose from TF, or None if unavailable or invalid."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, frame_id, Time(),
            ).transform
            return ReferencePose(
                (tf.rotation.x, tf.rotation.y,
                 tf.rotation.z, tf.rotation.w),
                (tf.translation.x, tf.translation.y,
                 tf.translation.z),
            )
        except (tf2_ros.TransformException, ValueError) as exc:
            self.get_logger().warn(
                f"No valid TF {self.world_frame} -> {frame_id} ({exc}); "
                f"boxes use the cloud frame, no height band",
                throttle_duration_sec=5.0,
            )
            return None

    def process(self, msg):
        """Returns (points, ObjectFrame) for one input message."""
        points = cloud_to_xyz(msg)
        pose = self.lookup_pose(msg.header.frame_id)
        gravity_aligned = pose is not None
        if not gravity_aligned:
            self.no_tf_frames += 1
            pose = IDENTITY_POSE

        start = time.perf_counter()
        if gravity_aligned:
            band = height_band_mask(
                points, pose,
                self.ground_z + self.crop[0], self.ground_z + self.crop[1],
            )
            points = points[band]
        groups = self.extractor.extract(points)
        middle = time.perf_counter()
        objects, rejected = estimate_objects(
            points, groups, pose, self.estimator_params,
        )
        end = time.perf_counter()

        self.cluster_ms.append((middle - start) * 1e3)
        self.estimate_ms.append((end - middle) * 1e3)
        self.total_rejected += rejected

        frame = ObjectFrame(
            stamp_ns=stamp_to_ns(msg.header.stamp),
            frame_id=msg.header.frame_id,
            reference_frame=(
                self.world_frame if gravity_aligned else msg.header.frame_id
            ),
            gravity_aligned=gravity_aligned,
            box_orientation_xyzw=pose.box_orientation_xyzw,
            objects=objects,
        )
        return points, frame

    def callback(self, msg):
        start = time.perf_counter()
        self.received += 1

        try:
            points, frame = self.process(msg)
        except Exception as exc:
            # Keep the frame visible downstream as an empty result.
            self.failed += 1
            self.get_logger().error(f"Object estimation failed: {exc}")
            points = np.empty((0, 3), dtype=np.float32)
            frame = ObjectFrame(
                stamp_ns=stamp_to_ns(msg.header.stamp),
                frame_id=msg.header.frame_id,
                reference_frame=msg.header.frame_id,
                gravity_aligned=False,
                box_orientation_xyzw=IDENTITY_POSE.box_orientation_xyzw,
                objects=[],
            )

        self.object_publisher.publish(String(data=frame_to_json(frame)))
        self.marker_publisher.publish(frame_to_markers(
            msg.header, frame.objects, show_labels=self.show_labels,
        ))
        self.points_publisher.publish(
            labeled_cloud(msg.header, points, frame.objects)
        )

        self.published += 1
        self.total_objects += len(frame.objects)
        elapsed_ms = (time.perf_counter() - start) * 1e3
        self.total_ms.append(elapsed_ms)

        if self.published % 30 == 0:
            nearest = ", ".join(
                f"#{obj.id} {obj.distance:.1f}m "
                f"{np.round(obj.size, 1).tolist()} n={obj.point_count}"
                for obj in frame.objects[:3]
            )
            self.get_logger().info(
                f"frames={self.published} objects={len(frame.objects)} "
                f"nearest=[{nearest}] processing={elapsed_ms:.1f}ms"
            )

    @staticmethod
    def _timing(values):
        if not values:
            return "n/a"
        p50, p95 = np.percentile(values, [50, 95])
        return f"p50={p50:.1f} p95={p95:.1f} max={max(values):.1f}ms"

    def report(self):
        mean = self.total_objects / self.published if self.published else 0.0
        self.get_logger().info(
            f"Object stats: "
            f"received={self.received}, "
            f"published={self.published}, "
            f"failed={self.failed}, "
            f"no_tf_frames={self.no_tf_frames}, "
            f"objects_per_frame={mean:.1f}, "
            f"rejected={self.total_rejected}, "
            f"cluster {self._timing(self.cluster_ms)}, "
            f"estimate {self._timing(self.estimate_ms)}, "
            f"total {self._timing(self.total_ms)}"
        )


def main():
    rclpy.init()
    node = None

    try:
        node = LidarObjectNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        # A signal can invalidate the context while spin creates its wait
        # set. That is a normal shutdown; anything else is a real error.
        if rclpy.ok():
            raise
    finally:
        if node is not None:
            node.report()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

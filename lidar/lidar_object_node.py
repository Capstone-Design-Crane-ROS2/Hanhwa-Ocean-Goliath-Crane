#!/usr/bin/env python3
"""ROS2 LiDAR-only object estimation node.

Input:  /lidar/points_preprocessed   PointCloud2
Output: /lidar/objects               String JSON, one per input frame
        /lidar/objects/markers       MarkerArray for RViz2
        /lidar/objects/points        PointCloud2 with an object_id field

Objects are tracked across frames (object_tracker.py, enabled by default):
each JSON object gets a track_id that stays the same for one physical
object, and markers keep their color per track. The frame-local id and the
object_id field of the points cloud are unchanged.
tracker.enabled:=false restores the untracked output exactly.

Velocity and crane-face TTC (object_motion.py, crane_boundary.py, enabled
with tracking by default): each object gets "motion" (Kalman velocity with
a validity state) and "crane_boundary" (box distance to the crane face
plane, closing speed, TTC or the reason there is none); the top level gets
"motion_model", "crane_boundary_model" and "predicted_tracks" (confirmed
tracks not detected in this frame, predicted only). The TTC is a
first-order approach-risk indicator: plane boundary, crane assumed static.
motion.enabled:=false restores the tracking-only output exactly.

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
from crane_boundary import CraneState, PlaneBoundary, zones_from_flat
from object_messages import frame_to_json, frame_to_markers, labeled_cloud
from object_motion import (
    MotionEstimator,
    MotionParams,
    TtcParams,
    assess_boundary,
    model_info,
    motion_to_dict,
    risk_to_dict,
)
from object_tracker import ObjectTracker, TrackerParams


# Height band above ground: keeps the tallest GT object (9.19 m) and drops
# roofs and upper walls, about half of the points on the S2/S4/S7 scenes.
DEFAULT_CROP_MIN_HEIGHT = -1.0
DEFAULT_CROP_MAX_HEIGHT = 10.0

TRACKER_PARAMS = (
    "accel_std", "meas_std_xy", "meas_std_z", "init_vel_std", "gate_chi2",
    "max_gate_distance", "min_hits", "max_coast_s", "tentative_coast_s",
    "max_gap_s",
)

MOTION_PARAMS = tuple(MotionParams.__dataclass_fields__)
TTC_PARAMS = tuple(TtcParams.__dataclass_fields__)
MOTION_INT_PARAMS = ("min_hits", "point_jump_min")

# Crane contact face of the dataset (gt.json eval_frame, face "C": world x
# = ox, outward +x). Scenario values, not a crane specification: set them
# for the real crane.
DEFAULT_FACE_POINT = (398.884, 0.052)
DEFAULT_FACE_NORMAL = (1.0, 0.0)
# Crane's own structure seen by the LiDAR in every bag: clusters 2-8 m
# behind the face at 6-9 m height (incl. a 2.6 x 4.1 x 1.1 m part 2.7 m from
# the sensor). Objects centered here are "excluded", not intrusions.
# World [x0, y0, z0, x1, y1, z1]; dataset-derived like the face.
DEFAULT_EXCLUSION_BOXES = [388.884, -35.0, 4.5, 398.884, 35.0, 12.0]

STARTUP_ONLY = (
    "input_topic", "output_topic", "marker_topic", "points_topic",
    "world_frame", "ground_z_world",
)


def _non_negative(value, name):
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and >= 0")
    return value


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
        self.declare_parameter("tracker.enabled", True)
        for name in TRACKER_PARAMS:
            self.declare_parameter(
                f"tracker.{name}", getattr(TrackerParams, name)
            )
        self.declare_parameter("motion.enabled", True)
        for name in MOTION_PARAMS:
            self.declare_parameter(
                f"motion.{name}", getattr(MotionParams, name)
            )
        for name in TTC_PARAMS:
            self.declare_parameter(f"ttc.{name}", getattr(TtcParams, name))
        self.declare_parameter("crane.face_point_x", DEFAULT_FACE_POINT[0])
        self.declare_parameter("crane.face_point_y", DEFAULT_FACE_POINT[1])
        self.declare_parameter("crane.face_normal_x", DEFAULT_FACE_NORMAL[0])
        self.declare_parameter("crane.face_normal_y", DEFAULT_FACE_NORMAL[1])
        self.declare_parameter("crane.velocity_x", 0.0)
        self.declare_parameter("crane.velocity_y", 0.0)
        self.declare_parameter("crane.exclusion_enabled", True)
        self.declare_parameter("crane.exclusion_boxes",
                               DEFAULT_EXCLUSION_BOXES)
        self.declare_parameter("marker.show_motion", True)
        self.declare_parameter("marker.velocity_arrow_s", 1.0)

        self.tracker = None
        self.motion = None
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
        self.track_ms = []
        self.motion_ms = []
        self.total_ms = []
        self.ttc_status = {}
        self.motion_states = {}

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
            "tracking": bool(value("tracker.enabled")),
            "tracker": TrackerParams(**{
                name: (int if name == "min_hits" else float)(
                    value(f"tracker.{name}")
                )
                for name in TRACKER_PARAMS
            }),
            "motion_enabled": bool(value("motion.enabled")),
            "motion": MotionParams(**{
                name: (int if name in MOTION_INT_PARAMS else float)(
                    value(f"motion.{name}")
                )
                for name in MOTION_PARAMS
            }),
            "ttc": TtcParams(**{
                name: float(value(f"ttc.{name}")) for name in TTC_PARAMS
            }),
            "boundary": PlaneBoundary(
                (float(value("crane.face_point_x")),
                 float(value("crane.face_point_y"))),
                (float(value("crane.face_normal_x")),
                 float(value("crane.face_normal_y"))),
            ),
            "crane": CraneState((float(value("crane.velocity_x")),
                                 float(value("crane.velocity_y")))),
            "zones": (zones_from_flat(value("crane.exclusion_boxes"))
                      if bool(value("crane.exclusion_enabled")) else []),
            "show_motion": bool(value("marker.show_motion")),
            "arrow_s": _non_negative(float(value("marker.velocity_arrow_s")),
                                     "marker.velocity_arrow_s"),
        }

    def _apply_settings(self, settings):
        self.extractor = EuclideanClusterExtractor(settings["cluster"])
        self.estimator_params = settings["estimate"]
        self.crop = settings["crop"]
        self.show_labels = settings["show_labels"]
        if not settings["tracking"]:
            self.tracker = None
        elif self.tracker is None:
            self.tracker = ObjectTracker(settings["tracker"])
        else:
            # Keeps the current tracks; new values apply from the next frame.
            self.tracker.params = settings["tracker"]
        # Motion needs tracks; a new tracker also starts a new estimator.
        if self.tracker is None or not settings["motion_enabled"]:
            self.motion = None
        elif self.motion is None or self.motion_tracker is not self.tracker:
            self.motion = MotionEstimator(settings["motion"])
        else:
            self.motion.params = settings["motion"]
        self.motion_tracker = self.tracker
        self.ttc_params = settings["ttc"]
        self.boundary = settings["boundary"]
        self.crane = settings["crane"]
        self.zones = settings["zones"]
        self.show_motion = settings["show_motion"]
        self.arrow_s = settings["arrow_s"]

    def _settings_text(self):
        cluster = self.extractor.params
        estimate = self.estimator_params
        return (
            f"tolerance={cluster.tolerance}m "
            f"points=[{cluster.min_points}, {cluster.max_points}] "
            f"crop={self.crop}m center={estimate.center_method} "
            f"max_footprint={estimate.max_footprint}m "
            f"max_height={estimate.max_height}m "
            f"tracking={self._tracker_text()} "
            f"motion={self._motion_text()}"
        )

    def _motion_text(self):
        if self.motion is None:
            return "off"
        face = self.boundary.describe()
        return (
            f"on(face={face['point_xy']} normal={face['normal_xy']} "
            f"margin={self.ttc_params.safety_margin_m}m "
            f"max_ttc={self.ttc_params.max_ttc_s}s "
            f"exclusion_zones={len(self.zones)} "
            f"crane={'static' if self.crane.static else 'moving'})"
        )

    def _tracker_text(self):
        if self.tracker is None:
            return "off"
        params = self.tracker.params
        return (
            f"on(min_hits={params.min_hits} "
            f"max_coast={params.max_coast_s}s "
            f"gate={params.max_gate_distance}m)"
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
        """Returns (points, ObjectFrame, track_ids, motion) for one message.

        track_ids is None when tracking is disabled; motion is None when
        motion is disabled, else (object_extras, frame_extras,
        motion_markers).
        """
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

        track_ids = None
        if self.tracker is not None:
            start = time.perf_counter()
            track_ids = self.tracker.update(
                frame.stamp_ns,
                [obj.center_reference for obj in objects],
                frame.reference_frame,
            )
            self.track_ms.append((time.perf_counter() - start) * 1e3)

        motion = None
        if self.motion is not None:
            start = time.perf_counter()
            motion = self.motion_outputs(frame, pose)
            self.motion_ms.append((time.perf_counter() - start) * 1e3)
        return points, frame, track_ids, motion

    def motion_outputs(self, frame, pose):
        """Velocity validity, boundary distance and TTC of a tracked frame.

        Returns (object_extras, frame_extras, motion_markers).
        """
        world = frame.gravity_aligned
        stamp = frame.stamp_ns
        motions = self.motion.update(self.tracker, stamp, frame.objects,
                                     world)
        extras, markers = [], []
        for obj, motion in zip(frame.objects, motions):
            risk = assess_boundary(
                motion, obj.bbox_min, obj.bbox_max, obj.center_reference,
                self.boundary, self.ttc_params, self.crane, self.zones,
                world,
            )
            self._count(motion.state, risk.status)
            extras.append({"motion": motion_to_dict(motion),
                           "crane_boundary": risk_to_dict(risk)})
            markers.append(self._motion_marker(obj, motion, risk, pose))

        predicted = []
        for track, motion in self.motion.predicted(self.tracker, stamp,
                                                   world):
            center = track.position
            half = self.motion.last_size(track.id)
            half = np.zeros(2) if half is None else half / 2.0
            box_min = np.concatenate([center[:2] - half, center[2:]])
            box_max = np.concatenate([center[:2] + half, center[2:]])
            risk = assess_boundary(
                motion, box_min, box_max, center, self.boundary,
                self.ttc_params, self.crane, self.zones, world,
            )
            predicted.append({
                "track_id": track.id,
                "center_reference": np.round(center, 3).tolist(),
                "since_update_s": round(motion.since_update_s, 3),
                "motion": motion_to_dict(motion),
                "crane_boundary": risk_to_dict(risk),
            })
        frame_extras = model_info(self.boundary, self.motion.params,
                                  self.ttc_params, self.crane, self.zones,
                                  frame.reference_frame)
        frame_extras["predicted_tracks"] = predicted
        return extras, frame_extras, markers

    def _count(self, state, status):
        self.motion_states[state] = self.motion_states.get(state, 0) + 1
        self.ttc_status[status] = self.ttc_status.get(status, 0) + 1

    def _motion_marker(self, obj, motion, risk, pose):
        """Arrow and text for moving objects (and any valid TTC)."""
        if not self.show_motion or motion.state != "moving":
            return None
        velocity = np.array([motion.velocity[0], motion.velocity[1], 0.0])
        tip = obj.center_reference + velocity * self.arrow_s
        text = f"{motion.speed:.1f}m/s"
        if risk.status == "valid":
            text += f" TTC {risk.ttc:.1f}s"
        return {
            "arrow_end": (pose.to_cloud(tip[None])[0]
                          if self.arrow_s > 0 else None),
            "text": text,
            "alert": risk.status == "valid",
        }

    def callback(self, msg):
        start = time.perf_counter()
        self.received += 1

        try:
            points, frame, track_ids, motion = self.process(msg)
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
            # The tracker did not see this frame; the next real stamp
            # difference covers the gap.
            track_ids = None if self.tracker is None else []
            motion = None
            if self.motion is not None:
                info = model_info(self.boundary, self.motion.params,
                                  self.ttc_params, self.crane, self.zones,
                                  frame.reference_frame)
                info["predicted_tracks"] = []
                motion = ([], info, [])

        object_extras, frame_extras, motion_markers = motion or (
            None, None, None)
        self.object_publisher.publish(String(
            data=frame_to_json(frame, track_ids, object_extras, frame_extras)
        ))
        self.marker_publisher.publish(frame_to_markers(
            msg.header, frame.objects, show_labels=self.show_labels,
            track_ids=track_ids, motion_markers=motion_markers,
        ))
        self.points_publisher.publish(
            labeled_cloud(msg.header, points, frame.objects)
        )

        self.published += 1
        self.total_objects += len(frame.objects)
        elapsed_ms = (time.perf_counter() - start) * 1e3
        self.total_ms.append(elapsed_ms)

        if self.published % 30 == 0:
            tracks = track_ids or [None] * len(frame.objects)
            nearest = ", ".join(
                f"#{obj.id}"
                + (f"/T{track}" if track is not None else "")
                + f" {obj.distance:.1f}m "
                f"{np.round(obj.size, 1).tolist()} n={obj.point_count}"
                for obj, track in zip(frame.objects[:3], tracks)
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
            f"track {self._timing(self.track_ms)}, "
            f"motion {self._timing(self.motion_ms)}, "
            f"total {self._timing(self.total_ms)}"
        )
        if self.motion is not None:
            self.get_logger().info(
                f"Motion stats: states={self.motion_states}, "
                f"ttc_status={self.ttc_status}"
            )
        if self.tracker is not None:
            self.get_logger().info(
                f"Tracker stats: tracks_created={self.tracker.next_id}, "
                f"active={len(self.tracker.tracks)}, "
                f"confirmed={len(self.tracker.confirmed_tracks)}, "
                f"resets={self.tracker.resets}"
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

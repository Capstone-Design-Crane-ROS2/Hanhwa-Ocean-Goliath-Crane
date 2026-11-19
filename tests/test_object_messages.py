"""Unit tests for /lidar/objects message builders."""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "lidar"),
)

from builtin_interfaces.msg import Time
from std_msgs.msg import Header
from visualization_msgs.msg import Marker

from object_estimator import ObjectFrame, ReferencePose, estimate_objects
from object_messages import (
    BOX_EDGES,
    LABELED_DTYPE,
    OBJECT_SCHEMA,
    frame_to_json,
    frame_to_markers,
    labeled_cloud,
)


POSE = ReferencePose(
    (0.17364817766693036, 0.0, 0.9848077530122081, 0.0),
    (396.8843504349122, 0.052, 5.669106866204646),
)
HEADER = Header(stamp=Time(sec=12, nanosec=345), frame_id="OS1_A")


def make_objects():
    world = np.array([
        [410.0, 0.0, 0.2], [410.5, 0.5, 1.8], [410.2, 0.1, 1.0],
        [420.0, 5.0, 0.2], [421.0, 6.0, 2.0],
    ])
    points = POSE.to_cloud(world).astype(np.float32)
    groups = [np.array([0, 1, 2]), np.array([3, 4])]
    return points, estimate_objects(points, groups, POSE)[0]


def make_frame(objects):
    return ObjectFrame(
        stamp_ns=12_000_000_345,
        frame_id="OS1_A",
        reference_frame="world",
        gravity_aligned=True,
        box_orientation_xyzw=POSE.box_orientation_xyzw,
        objects=objects,
    )


class JsonTest(unittest.TestCase):
    def test_fields(self):
        _, objects = make_objects()
        data = json.loads(frame_to_json(make_frame(objects)))
        self.assertEqual(data["schema"], OBJECT_SCHEMA)
        self.assertEqual(data["stamp_ns"], 12_000_000_345)
        self.assertEqual(data["frame_id"], "OS1_A")
        self.assertEqual(data["object_count"], 2)
        first = data["objects"][0]
        self.assertEqual(first["id"], 0)
        self.assertEqual(first["point_count"], 3)
        self.assertEqual(np.shape(first["corners"]), (8, 3))
        np.testing.assert_allclose(
            first["size_m"], objects[0].size, atol=1e-3
        )
        self.assertAlmostEqual(
            first["distance_m"], objects[0].distance, places=3
        )

    def test_empty_frame(self):
        data = json.loads(frame_to_json(make_frame([])))
        self.assertEqual(data["object_count"], 0)
        self.assertEqual(data["objects"], [])


class LabeledCloudTest(unittest.TestCase):
    def test_points_and_labels(self):
        points, objects = make_objects()
        msg = labeled_cloud(HEADER, points, objects)
        self.assertEqual(msg.header, HEADER)
        self.assertEqual(msg.width, 5)
        self.assertEqual(msg.point_step, 16)
        self.assertEqual(len(msg.data), 5 * 16)

        data = np.frombuffer(bytes(msg.data), dtype=LABELED_DTYPE)
        self.assertEqual(data["object_id"].tolist(), [0, 0, 0, 1, 1])
        for obj in objects:
            selected = data[data["object_id"] == obj.id]
            np.testing.assert_array_equal(
                np.column_stack([selected["x"], selected["y"], selected["z"]]),
                points[obj.point_indices],
            )

    def test_empty(self):
        msg = labeled_cloud(HEADER, np.empty((0, 3)), [])
        self.assertEqual(msg.header.frame_id, "OS1_A")
        self.assertEqual(msg.width, 0)
        self.assertEqual(len(msg.data), 0)


class MarkerTest(unittest.TestCase):
    def test_structure(self):
        _, objects = make_objects()
        markers = frame_to_markers(HEADER, objects).markers
        self.assertEqual(markers[0].action, Marker.DELETEALL)
        self.assertEqual(len(markers), 1 + 2 * 2)
        for marker in markers:
            self.assertEqual(marker.header, HEADER)
        boxes = [m for m in markers if m.ns == "object_box"]
        labels = [m for m in markers if m.ns == "object_label"]
        self.assertEqual([m.id for m in boxes], [0, 1])
        self.assertEqual(labels[0].text,
                         f"#0 {objects[0].distance:.1f}m")
        for box in boxes:
            self.assertEqual(box.type, Marker.LINE_LIST)
            self.assertEqual(len(box.points), 2 * len(BOX_EDGES))

    def test_box_points_are_the_corners(self):
        _, objects = make_objects()
        box = frame_to_markers(HEADER, objects).markers[1]
        points = np.array([[p.x, p.y, p.z] for p in box.points])
        for point in points:
            self.assertLess(
                np.linalg.norm(objects[0].corners - point, axis=1).min(), 1e-9,
            )

    def test_edges_form_a_box(self):
        self.assertEqual(len(set(BOX_EDGES)), 12)
        counts = np.bincount(np.array(BOX_EDGES).ravel())
        self.assertEqual(counts.tolist(), [3] * 8)

    def test_empty_frame_clears_display(self):
        markers = frame_to_markers(HEADER, []).markers
        self.assertEqual(len(markers), 1)
        self.assertEqual(markers[0].action, Marker.DELETEALL)

    def test_labels_can_be_disabled(self):
        _, objects = make_objects()
        markers = frame_to_markers(HEADER, objects, show_labels=False).markers
        self.assertEqual(len(markers), 3)


class TrackIdTest(unittest.TestCase):
    def test_json_without_tracking_has_no_track_id(self):
        _, objects = make_objects()
        data = json.loads(frame_to_json(make_frame(objects)))
        for item in data["objects"]:
            self.assertNotIn("track_id", item)

    def test_json_track_ids_and_null(self):
        _, objects = make_objects()
        data = json.loads(frame_to_json(make_frame(objects), [7, None]))
        self.assertEqual([o["id"] for o in data["objects"]], [0, 1])
        self.assertEqual([o["track_id"] for o in data["objects"]],
                         [7, None])

    def test_json_rest_unchanged_by_tracking(self):
        _, objects = make_objects()
        plain = json.loads(frame_to_json(make_frame(objects)))
        tracked = json.loads(frame_to_json(make_frame(objects), [3, 4]))
        for item in tracked["objects"]:
            del item["track_id"]
        self.assertEqual(plain, tracked)

    def test_track_ids_length_checked(self):
        _, objects = make_objects()
        with self.assertRaises(ValueError):
            frame_to_json(make_frame(objects), [1])

    def test_markers_follow_track_id(self):
        _, objects = make_objects()
        plain = frame_to_markers(HEADER, objects).markers
        tracked = frame_to_markers(HEADER, objects, track_ids=[9, None])
        boxes = [m for m in tracked.markers if m.ns == "object_box"]
        labels = [m for m in tracked.markers if m.ns == "object_label"]
        # Marker IDs stay the frame-local ids.
        self.assertEqual([m.id for m in boxes], [0, 1])
        self.assertEqual(labels[0].text, f"T9 {objects[0].distance:.1f}m")
        self.assertEqual(labels[1].text, f"#1 {objects[1].distance:.1f}m")
        # Track 9 has the color object 9 would have; untracked is gray.
        reference = frame_to_markers(HEADER, objects, track_ids=[9, 9])
        self.assertEqual(boxes[0].color, reference.markers[1].color)
        self.assertEqual(boxes[1].color.r, boxes[1].color.g)
        self.assertNotEqual(boxes[0].color, plain[1].color)


if __name__ == "__main__":
    unittest.main()

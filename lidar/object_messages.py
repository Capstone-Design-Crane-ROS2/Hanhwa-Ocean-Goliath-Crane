"""ROS message builders for ObjectFrame results (no node, no subscriptions).

JSON string: schema lidar-objects-v1, documented in
docs/lidar_object_estimation.md.
"""

import array
import json

import numpy as np
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


OBJECT_SCHEMA = "lidar-objects-v1"

# Corner pairs of the 12 box edges (order from object_estimator.box_corners).
BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

PALETTE = (
    (0.90, 0.30, 0.25), (0.25, 0.70, 0.95), (0.35, 0.85, 0.40),
    (0.98, 0.75, 0.20), (0.75, 0.45, 0.95), (0.20, 0.85, 0.80),
    (0.98, 0.50, 0.70), (0.70, 0.85, 0.25),
)

LABELED_FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="object_id", offset=12,
               datatype=PointField.UINT32, count=1),
]
LABELED_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("object_id", "<u4"),
])


def _rounded(values, digits=3):
    return np.round(np.asarray(values, dtype=np.float64), digits).tolist()


def object_to_dict(obj):
    """JSON-ready dict of one LidarObject."""
    return {
        "id": obj.id,
        "point_count": obj.point_count,
        "center": _rounded(obj.center),
        "centroid": _rounded(obj.centroid),
        "bbox_center": _rounded(obj.bbox_center),
        "distance_m": round(obj.distance, 3),
        "horizontal_distance_m": round(obj.horizontal_distance, 3),
        "nearest_distance_m": round(obj.nearest_distance, 3),
        "size_m": _rounded(obj.size),
        "bbox_min_reference": _rounded(obj.bbox_min),
        "bbox_max_reference": _rounded(obj.bbox_max),
        "center_reference": _rounded(obj.center_reference),
        "corners": _rounded(obj.corners),
    }


def frame_to_json(frame):
    """ObjectFrame -> JSON text for /lidar/objects."""
    return json.dumps({
        "schema": OBJECT_SCHEMA,
        "stamp_ns": frame.stamp_ns,
        "frame_id": frame.frame_id,
        "reference_frame": frame.reference_frame,
        "gravity_aligned": frame.gravity_aligned,
        "box_orientation_xyzw": _rounded(frame.box_orientation_xyzw, 6),
        "object_count": len(frame.objects),
        "objects": [object_to_dict(obj) for obj in frame.objects],
    })


def labeled_cloud(header, points, objects):
    """PointCloud2 of all objects' points with an object_id field.

    Points are given in the cloud frame. header is reused unchanged, so the
    stamp and frame_id match the input cloud.
    """
    pts = np.asarray(points)
    count = sum(obj.point_count for obj in objects)
    data = np.empty(count, dtype=LABELED_DTYPE)
    start = 0
    for obj in objects:
        end = start + obj.point_count
        selected = pts[obj.point_indices]
        data["x"][start:end] = selected[:, 0]
        data["y"][start:end] = selected[:, 1]
        data["z"][start:end] = selected[:, 2]
        data["object_id"][start:end] = obj.id
        start = end

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = count
    msg.fields = LABELED_FIELDS
    msg.is_bigendian = False
    msg.point_step = LABELED_DTYPE.itemsize
    msg.row_step = count * LABELED_DTYPE.itemsize
    msg.is_dense = True
    # bytes assignment validates every byte in Python; see xyz_to_cloud.
    payload = array.array("B")
    payload.frombytes(data.tobytes())
    msg.data = payload
    return msg


def _color(object_id, alpha=1.0):
    r, g, b = PALETTE[object_id % len(PALETTE)]
    return ColorRGBA(r=r, g=g, b=b, a=alpha)


def _box_marker(header, obj, line_width):
    marker = Marker()
    marker.header = header
    marker.ns = "object_box"
    marker.id = obj.id
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = line_width
    marker.color = _color(obj.id)
    corners = obj.corners.tolist()
    marker.points = [
        Point(x=corners[i][0], y=corners[i][1], z=corners[i][2])
        for edge in BOX_EDGES for i in edge
    ]
    return marker


def _label_marker(header, obj, text_height):
    marker = Marker()
    marker.header = header
    marker.ns = "object_label"
    marker.id = obj.id
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD
    # Above the box, along the box's own vertical edge.
    top = obj.corners[4:].mean(axis=0)
    up = obj.corners[4] - obj.corners[0]
    length = np.linalg.norm(up)
    if length > 1e-6:
        top = top + up / length * text_height
    marker.pose.position.x = float(top[0])
    marker.pose.position.y = float(top[1])
    marker.pose.position.z = float(top[2])
    marker.pose.orientation.w = 1.0
    marker.scale.z = text_height
    marker.color = _color(obj.id)
    marker.text = f"#{obj.id} {obj.distance:.1f}m"
    return marker


def frame_to_markers(header, objects, line_width=0.05, text_height=0.6,
                     show_labels=True):
    """MarkerArray with one wireframe box and label per object.

    The first marker is DELETEALL, so boxes of the previous frame never
    remain when the object count drops; an empty frame clears the display.
    """
    clear = Marker()
    clear.header = header
    clear.action = Marker.DELETEALL

    markers = [clear]
    for obj in objects:
        markers.append(_box_marker(header, obj, line_width))
        if show_labels:
            markers.append(_label_marker(header, obj, text_height))
    return MarkerArray(markers=markers)

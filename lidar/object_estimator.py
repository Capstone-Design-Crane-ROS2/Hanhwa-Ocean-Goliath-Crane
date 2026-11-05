"""Object position, distance, size and box from a group of points.

Independent of how the points were grouped (see cluster_extractor.py) and
free of ROS, so LiDAR clusters and YOLO-selected points share this code.

The dataset's OS1_A z axis is tilted 20 deg from vertical. An axis-aligned
box in OS1_A would therefore not be level: a 1.8 m person would look about
0.6 m wider, and "height" would not be height. Boxes are axis-aligned in
the reference frame (world) given by ReferencePose instead, and reported
back in the cloud frame as 8 corners. Without a pose the cloud frame itself
is used and ObjectFrame.gravity_aligned is False.
"""

from dataclasses import dataclass, field, replace
from typing import List

import numpy as np


CENTER_METHODS = ("bbox", "centroid")


def _quaternion_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@dataclass(frozen=True)
class ReferencePose:
    """Reference <- cloud transform. Boxes are axis-aligned in reference."""

    rotation_xyzw: tuple = (0.0, 0.0, 0.0, 1.0)
    translation: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        q = np.asarray(self.rotation_xyzw, dtype=np.float64)
        t = np.asarray(self.translation, dtype=np.float64)
        if q.shape != (4,) or t.shape != (3,):
            raise ValueError("Expected quaternion (4,) and translation (3,)")
        if not (np.isfinite(q).all() and np.isfinite(t).all()):
            raise ValueError("Transform must be finite")
        norm = np.linalg.norm(q)
        if abs(norm - 1.0) > 1e-3:
            raise ValueError(f"Quaternion is not normalized (|q|={norm:.6f})")
        object.__setattr__(self, "_q", q / norm)
        object.__setattr__(self, "_t", t)
        object.__setattr__(self, "_r", _quaternion_to_matrix(q / norm))

    def to_reference(self, points):
        """Cloud frame -> reference frame, for (N, 3) points."""
        return np.asarray(points, dtype=np.float64) @ self._r.T + self._t

    def to_cloud(self, points):
        """Reference frame -> cloud frame, for (N, 3) points."""
        return (np.asarray(points, dtype=np.float64) - self._t) @ self._r

    @property
    def box_orientation_xyzw(self):
        """Orientation of the reference axes in the cloud frame."""
        x, y, z, w = self._q
        return (-x, -y, -z, w)


IDENTITY_POSE = ReferencePose()


@dataclass(frozen=True)
class EstimatorParams:
    """Object estimation parameters (metres)."""

    center_method: str = "bbox"
    # Candidates beyond these box limits are structure, not objects.
    # 0 disables a limit.
    max_footprint: float = 15.0
    max_height: float = 0.0

    def __post_init__(self):
        if self.center_method not in CENTER_METHODS:
            raise ValueError(f"center_method must be one of {CENTER_METHODS}")
        for name in ("max_footprint", "max_height"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")


@dataclass(frozen=True, eq=False)
class LidarObject:
    """One object candidate. Vectors are float64 and in metres.

    Cloud frame (header.frame_id, origin = sensor):
        center, centroid, bbox_center, corners, distance values.
    Reference frame (world when gravity-aligned):
        bbox_min, bbox_max, size, center_reference.
    """

    id: int
    point_indices: np.ndarray    # indices into the cloud given to estimate
    point_count: int
    center: np.ndarray           # bbox_center or centroid, see center_method
    centroid: np.ndarray         # mean of the points
    bbox_center: np.ndarray      # center of the box
    distance: float              # |center|, straight line from the sensor
    horizontal_distance: float   # same, ignoring reference z
    nearest_distance: float      # closest point of the object to the sensor
    size: np.ndarray             # box edge lengths: x, y, z (= height)
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    center_reference: np.ndarray
    corners: np.ndarray          # (8, 3): bottom 4 CCW, then top 4


@dataclass(frozen=True, eq=False)
class ObjectFrame:
    """All objects of one cloud, with the cloud's stamp and frame."""

    stamp_ns: int
    frame_id: str
    reference_frame: str
    gravity_aligned: bool
    box_orientation_xyzw: tuple
    objects: List[LidarObject] = field(default_factory=list)


def box_corners(bbox_min, bbox_max):
    """8 box corners: bottom 4 CCW seen from above, then top 4."""
    (x0, y0, z0), (x1, y1, z1) = bbox_min, bbox_max
    return np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float64)


def estimate_object(points, indices, object_id, pose=IDENTITY_POSE,
                    center_method="bbox"):
    """Estimate one object from the given point indices.

    Args:
        points: (N, 3) cloud in the cloud frame (sensor at the origin).
        indices: Non-empty index array selecting the object's points.
        object_id: Identifier stored in the result.
        pose: ReferencePose defining the box axes.
        center_method: "bbox" or "centroid"; selects LidarObject.center.

    Returns:
        LidarObject.
    """
    if center_method not in CENTER_METHODS:
        raise ValueError(f"center_method must be one of {CENTER_METHODS}")
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        raise ValueError("An object needs at least one point")

    cloud_pts = np.asarray(points, dtype=np.float64)[indices]
    reference_pts = pose.to_reference(cloud_pts)

    bbox_min = reference_pts.min(axis=0)
    bbox_max = reference_pts.max(axis=0)
    bbox_center_reference = (bbox_min + bbox_max) / 2.0

    centroid = cloud_pts.mean(axis=0)
    bbox_center = pose.to_cloud(bbox_center_reference[None])[0]
    if center_method == "bbox":
        center = bbox_center
        center_reference = bbox_center_reference
    else:
        center = centroid
        center_reference = pose.to_reference(centroid[None])[0]

    # Horizontal = perpendicular to the reference z axis, through the sensor.
    offset = center_reference - pose.to_reference(np.zeros((1, 3)))[0]

    return LidarObject(
        id=int(object_id),
        point_indices=indices,
        point_count=int(indices.size),
        center=center,
        centroid=centroid,
        bbox_center=bbox_center,
        distance=float(np.linalg.norm(center)),
        horizontal_distance=float(np.linalg.norm(offset[:2])),
        nearest_distance=float(np.linalg.norm(cloud_pts, axis=1).min()),
        size=bbox_max - bbox_min,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        center_reference=center_reference,
        corners=pose.to_cloud(box_corners(bbox_min, bbox_max)),
    )


def height_band_mask(points, pose, min_z, max_z):
    """Points whose reference z is inside [min_z, max_z]. Non-finite: False."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return np.zeros(len(pts), dtype=bool)
    z = pose.to_reference(pts)[:, 2]
    with np.errstate(invalid="ignore"):
        return (z >= min_z) & (z <= max_z)


def estimate_objects(points, groups, pose=IDENTITY_POSE, params=None):
    """Estimate a frame's objects from point groups from any extractor.

    Candidates over the footprint/height limits are dropped. The rest get
    frame-local IDs 0..n-1 by ascending distance. IDs are not tracked: the
    same physical object can have a different ID in the next frame.

    Returns:
        (objects, rejected): list of LidarObject, number of dropped groups.
    """
    params = params or EstimatorParams()
    estimated = [
        estimate_object(points, group, 0, pose, params.center_method)
        for group in groups
    ]
    kept = [obj for obj in estimated if _within_limits(obj, params)]
    kept.sort(key=lambda obj: obj.distance)
    objects = [
        replace(obj, id=object_id) for object_id, obj in enumerate(kept)
    ]
    return objects, len(estimated) - len(objects)


def _within_limits(obj, params):
    if params.max_footprint > 0 and obj.size[:2].max() > params.max_footprint:
        return False
    if params.max_height > 0 and obj.size[2] > params.max_height:
        return False
    return True

"""Crane collision boundary models (ROS-free).

A boundary answers two questions for an object's world box:
    distance(bbox_min, bbox_max)  signed horizontal distance from the
                                  boundary to the nearest box point, > 0
                                  outside (free side), <= 0 touching or
                                  inside the boundary
    normal(bbox_min, bbox_max)    unit horizontal vector pointing away from
                                  the boundary at that nearest point; the
                                  closing speed is -normal . v_relative

Only PlaneBoundary exists now: the crane contact face as a vertical plane of
unlimited width, the only crane geometry in the dataset (gt.json eval_frame:
x_eval = world_x - ox is the distance out from the face). This is a first
approach-risk indicator, not a collision prediction: the real crane outline,
its height and its motion are unknown.

Extension points:
    - Another outline (2D polygon, 3D box) implements the same two methods
      and describe(); object_motion.py only uses this interface.
    - Crane motion: CraneState.velocity_xy is subtracted from the object
      velocity (relative velocity). It is zero (crane assumed static) until
      a crane motion source (rail encoder, odometry) exists. A moving crane
      also moves the sensor; then the world-frame object velocity itself
      needs that ego-motion (TF that is not static).
"""

from dataclasses import dataclass

import numpy as np


def box_corners_xy(bbox_min, bbox_max):
    """4 horizontal corners of an axis-aligned world box, (4, 2)."""
    x0, y0 = float(bbox_min[0]), float(bbox_min[1])
    x1, y1 = float(bbox_max[0]), float(bbox_max[1])
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


@dataclass(frozen=True)
class PlaneBoundary:
    """Vertical plane through point_xy with outward normal normal_xy.

    Outward = away from the crane, toward where objects come from. The
    plane has no width limit: an object far to the side still counts.
    """

    point_xy: tuple = (398.884, 0.052)
    normal_xy: tuple = (1.0, 0.0)

    def __post_init__(self):
        p = np.asarray(self.point_xy, dtype=np.float64)
        n = np.asarray(self.normal_xy, dtype=np.float64)
        if p.shape != (2,) or n.shape != (2,):
            raise ValueError("point_xy and normal_xy need 2 values")
        if not (np.isfinite(p).all() and np.isfinite(n).all()):
            raise ValueError("boundary values must be finite")
        length = np.linalg.norm(n)
        if length < 1e-9:
            raise ValueError("normal_xy must not be zero")
        object.__setattr__(self, "_p", p)
        object.__setattr__(self, "_n", n / length)

    def distance(self, bbox_min, bbox_max):
        offsets = (box_corners_xy(bbox_min, bbox_max) - self._p) @ self._n
        return float(offsets.min())

    def normal(self, bbox_min=None, bbox_max=None):
        return self._n.copy()

    @property
    def point(self):
        return self._p.copy()

    def describe(self):
        return {
            "type": "plane",
            "point_xy": np.round(self._p, 3).tolist(),
            "normal_xy": np.round(self._n, 6).tolist(),
            "width": "unlimited",
        }


@dataclass(frozen=True)
class CraneState:
    """Crane motion in the reference frame. Static until a source exists."""

    velocity_xy: tuple = (0.0, 0.0)

    def __post_init__(self):
        v = np.asarray(self.velocity_xy, dtype=np.float64)
        if v.shape != (2,) or not np.isfinite(v).all():
            raise ValueError("velocity_xy needs 2 finite values")
        object.__setattr__(self, "_v", v)

    @property
    def velocity(self):
        return self._v.copy()

    @property
    def static(self):
        return not np.any(self._v)


@dataclass(frozen=True)
class ExclusionZone:
    """World axis-aligned box. Objects whose center is inside are excluded
    from boundary risk (e.g. the crane's own structure seen by the LiDAR)."""

    box_min: tuple
    box_max: tuple

    def __post_init__(self):
        lo = np.asarray(self.box_min, dtype=np.float64)
        hi = np.asarray(self.box_max, dtype=np.float64)
        if lo.shape != (3,) or hi.shape != (3,):
            raise ValueError("exclusion box needs 3 + 3 values")
        if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
            raise ValueError("exclusion box must be finite")
        if np.any(lo > hi):
            raise ValueError("exclusion box min must be <= max")
        object.__setattr__(self, "_lo", lo)
        object.__setattr__(self, "_hi", hi)

    def contains(self, point):
        p = np.asarray(point, dtype=np.float64)
        return bool(np.all(p >= self._lo) and np.all(p <= self._hi))

    def describe(self):
        return {"min": np.round(self._lo, 3).tolist(),
                "max": np.round(self._hi, 3).tolist()}


def zones_from_flat(values):
    """[x0, y0, z0, x1, y1, z1, ...] -> list of ExclusionZone."""
    values = [float(v) for v in values]
    if len(values) % 6:
        raise ValueError("exclusion boxes need 6 values each")
    return [ExclusionZone(tuple(values[k:k + 3]), tuple(values[k + 3:k + 6]))
            for k in range(0, len(values), 6)]

"""Object velocity validity and crane-boundary TTC (ROS-free).

Velocity:
    The tracker's Kalman velocity (object_tracker.py) is the estimate; this
    module only decides whether it can be trusted. Horizontal (x, y) only.
    Each object gets a motion state:
        moving      valid velocity, speed >= stationary_speed
        stationary  valid velocity, speed <  stationary_speed
        unknown     not trusted; reason says why:
            unconfirmed_track  track not confirmed (no track_id yet)
            not_world_frame    positions are not in the world frame (no TF)
            initializing       track younger than min_track_age_s or fewer
                               than min_hits updates (new or after reset /
                               ID change: a new track ID starts over)
            shape_jump         box footprint or point count jumped within
                               the last anomaly_hold_s (split / merged
                               cluster: the center jumps, the Kalman
                               velocity follows)
            shape_changing     a footprint edge varied by more than
                               max(shape_change_ratio * edge,
                               shape_change_min_m) within the last window_s
                               (object gradually emerging from occlusion:
                               the box center moves slower than the object,
                               S7 truck next to the wall)
            high_uncertainty   1-sigma horizontal speed > max_speed_std
            implausible_speed  speed > max_speed
            inconsistent       Kalman velocity differs from the least-squares
                               velocity of the measured centers over the
                               last window_s by more than
                               max_window_mismatch, or the Kalman speed says
                               moving while the window speed is below
                               stationary_speed (Kalman still decaying from
                               a jump: moving needs both to agree)
    The window velocity (finite difference over a window, not two frames) is
    the cross-check; the published velocity is always the Kalman one.

TTC (first approach-risk indicator, plane boundary, crane assumed static):
    distance  = signed horizontal distance from the boundary to the nearest
                point of the object's world box (not the center, not the
                sensor distance)
    closing   = -normal . (v_object - v_crane), > 0 means approaching
    TTC       = (distance - safety_margin) / closing, assuming the current
                velocity is kept. Only computed when the velocity is valid
                and the object approaches; otherwise ttc is None and
                ttc_status says why. A range [d / (c + k*sigma_c),
                d / (c - k*sigma_c)] reflects the velocity uncertainty
                (upper end None when c - k*sigma_c <= 0).
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from crane_boundary import CraneState


MOTION_STATES = ("unknown", "stationary", "moving")
TTC_STATUSES = (
    "valid", "beyond_range", "not_approaching", "velocity_unknown",
    "inside_boundary", "within_margin", "excluded", "not_world_frame",
)
INDICATOR_NOTE = (
    "first-order approach-risk indicator: plane boundary, crane assumed "
    "static, current velocity assumed constant; not a collision prediction"
)


@dataclass(frozen=True)
class MotionParams:
    """Velocity validity thresholds (metres, seconds). Not safety limits."""

    min_track_age_s: float = 0.5
    min_hits: int = 5
    max_speed_std: float = 0.6
    stationary_speed: float = 0.3
    max_speed: float = 15.0
    # Footprint edge jump: |change| > max(ratio * previous, min_m).
    size_jump_ratio: float = 0.3
    size_jump_min_m: float = 0.5
    # Point count jump: |change| > max(ratio * larger, min_points).
    point_jump_ratio: float = 0.5
    point_jump_min: int = 10
    anomaly_hold_s: float = 1.0
    # Footprint edge range over window_s: > max(ratio * edge, min_m).
    shape_change_ratio: float = 0.3
    shape_change_min_m: float = 0.5
    window_s: float = 1.0
    max_window_mismatch: float = 0.3

    def __post_init__(self):
        for name in ("min_track_age_s", "max_speed_std", "stationary_speed",
                     "max_speed", "size_jump_ratio", "size_jump_min_m",
                     "point_jump_ratio", "anomaly_hold_s",
                     "shape_change_ratio", "shape_change_min_m", "window_s",
                     "max_window_mismatch"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        for name in ("max_speed_std", "max_speed", "window_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        for name in ("min_hits", "point_jump_min"):
            value = getattr(self, name)
            if int(value) != value or value < 0:
                raise ValueError(f"{name} must be an integer >= 0")


@dataclass(frozen=True)
class TtcParams:
    """TTC settings. Defaults are starting values, not safety criteria."""

    safety_margin_m: float = 0.0
    max_ttc_s: float = 10.0
    min_closing_speed: float = 0.3
    # Kalman velocity sigma is conservative (S7: sigma p50 0.43 m/s, actual
    # error p50 0.05 m/s), so 1 sigma is used for the approach test/range.
    sigma_k: float = 1.0

    def __post_init__(self):
        for name in ("safety_margin_m", "max_ttc_s", "min_closing_speed",
                     "sigma_k"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.max_ttc_s <= 0:
            raise ValueError("max_ttc_s must be > 0")


@dataclass(frozen=True, eq=False)
class ObjectMotion:
    """Velocity result of one object or predicted track (reference frame)."""

    track_id: Optional[int]
    state: str
    reason: Optional[str]
    measured: bool                   # False: track only predicted (coasting)
    velocity: Optional[np.ndarray]   # (2,) Kalman vx, vy; None without track
    velocity_cov: Optional[np.ndarray]  # (2, 2)
    speed: Optional[float]
    speed_std: Optional[float]       # 1-sigma, largest axis
    window_velocity: Optional[np.ndarray]  # (2,) cross-check, may be None
    track_age_s: Optional[float]
    since_update_s: Optional[float]

    @property
    def valid(self):
        return self.state != "unknown"


@dataclass(frozen=True, eq=False)
class BoundaryRisk:
    """Crane-boundary relation of one object."""

    distance: Optional[float]        # signed, box to boundary
    closing_speed: Optional[float]   # > 0 approaching
    closing_speed_std: Optional[float]
    direction: str                   # approaching / receding / steady /
    #                                  unknown
    ttc: Optional[float]
    ttc_range: Optional[tuple]       # (low, high or None)
    status: str


@dataclass
class _History:
    samples: deque = field(default_factory=deque)  # (stamp_ns, xy, size)
    size_xy: Optional[np.ndarray] = None
    point_count: Optional[int] = None
    anomaly_until_ns: int = -1
    last_stamp_ns: int = -1


def _window_velocity(samples, stamp_ns, window_s):
    """Least-squares xy velocity of samples inside the window, or None."""
    t0 = stamp_ns - int(window_s * 1e9)
    pts = [(s, xy) for s, xy, _ in samples if s >= t0]
    if len(pts) < 3:
        return None
    t = np.array([(s - stamp_ns) * 1e-9 for s, _ in pts])
    if t.max() - t.min() < window_s / 2:
        return None
    xy = np.array([xy for _, xy in pts])
    A = np.stack([t, np.ones_like(t)], axis=1)
    coeffs, *_ = np.linalg.lstsq(A, xy, rcond=None)
    return coeffs[0]


class MotionEstimator:
    """Per-frame velocity validity on top of an ObjectTracker."""

    def __init__(self, params=None):
        self.params = params or MotionParams()
        self.histories = {}
        self._tracker_resets = None
        self._last_stamp_ns = None

    def reset(self):
        self.histories = {}
        self._last_stamp_ns = None

    def update(self, tracker, stamp_ns, objects, world=True):
        """Motion of each object of the frame tracker.update() just saw.

        Args:
            tracker: the ObjectTracker, right after update(stamp_ns, ...).
            stamp_ns: frame stamp.
            objects: the frame's LidarObjects, same order as given to the
                tracker (center_reference, size, point_count are used).
            world: False when positions are not in the world frame.

        Returns:
            list of ObjectMotion, one per object.
        """
        stamp_ns = int(stamp_ns)
        if (self._tracker_resets != tracker.resets
                or (self._last_stamp_ns is not None
                    and stamp_ns < self._last_stamp_ns)):
            self.reset()
        self._tracker_resets = tracker.resets
        self._last_stamp_ns = stamp_ns

        tracks = tracker.last_tracks
        if len(tracks) != len(objects):
            raise ValueError("objects do not match the tracker's last update")
        live = set(tracker.tracks)
        for tid in list(self.histories):
            if tid not in live:
                del self.histories[tid]

        results = []
        for obj, track in zip(objects, tracks):
            if track is None:
                results.append(self._unknown(None, "unconfirmed_track"))
                continue
            if track.last_update_ns == stamp_ns:
                self._observe(track.id, stamp_ns, obj)
            if not track.confirmed:
                # Like track_id: nothing is reported before confirmation.
                results.append(self._unknown(None, "unconfirmed_track"))
                continue
            results.append(self._judge(track, stamp_ns, world, True))
        return results

    def predicted(self, tracker, stamp_ns, world=True):
        """Motion of confirmed tracks that were only predicted this frame.

        Returns:
            list of (Track, ObjectMotion) with measured=False.
        """
        return [(track, self._judge(track, int(stamp_ns), world, False))
                for track in tracker.coasting_tracks()]

    def last_size(self, track_id):
        history = self.histories.get(track_id)
        return None if history is None else history.size_xy

    def _observe(self, track_id, stamp_ns, obj):
        p = self.params
        history = self.histories.setdefault(track_id, _History())
        if history.last_stamp_ns == stamp_ns:
            return
        history.last_stamp_ns = stamp_ns
        size = np.asarray(obj.size, dtype=np.float64)[:2]
        count = int(obj.point_count)
        if history.size_xy is not None:
            change = np.abs(size - history.size_xy)
            limit = np.maximum(p.size_jump_ratio * history.size_xy,
                               p.size_jump_min_m)
            size_jump = bool(np.any(change > limit))
            point_change = abs(count - history.point_count)
            point_jump = point_change > max(
                p.point_jump_ratio * max(count, history.point_count),
                p.point_jump_min,
            )
            if size_jump or point_jump:
                history.anomaly_until_ns = stamp_ns + int(
                    p.anomaly_hold_s * 1e9
                )
        history.size_xy = size
        history.point_count = count
        history.samples.append(
            (stamp_ns, np.asarray(obj.center_reference, np.float64)[:2],
             size)
        )
        oldest = stamp_ns - int(p.window_s * 1e9)
        while history.samples and history.samples[0][0] < oldest:
            history.samples.popleft()

    def _shape_changing(self, history):
        p = self.params
        sizes = np.array([size for _, _, size in history.samples])
        if len(sizes) < 2:
            return False
        spread = sizes.max(axis=0) - sizes.min(axis=0)
        limit = np.maximum(p.shape_change_ratio * np.median(sizes, axis=0),
                           p.shape_change_min_m)
        return bool(np.any(spread > limit))

    def _unknown(self, track, reason, stamp_ns=None, measured=True):
        if track is None:
            return ObjectMotion(None, "unknown", reason, measured, None, None,
                                None, None, None, None, None)
        v = track.velocity[:2]
        cov = track.velocity_covariance[:2, :2]
        return ObjectMotion(
            track.id if track.confirmed else None, "unknown", reason,
            measured, v, cov, float(np.linalg.norm(v)),
            float(np.sqrt(np.linalg.eigvalsh(cov).max())), None,
            track.age_s(stamp_ns), track.since_update_s(stamp_ns),
        )

    def _judge(self, track, stamp_ns, world, measured):
        p = self.params
        v = track.velocity[:2]
        cov = track.velocity_covariance[:2, :2]
        speed = float(np.linalg.norm(v))
        speed_std = float(np.sqrt(max(np.linalg.eigvalsh(cov).max(), 0.0)))
        history = self.histories.get(track.id)
        window_v = (None if history is None else
                    _window_velocity(history.samples, stamp_ns, p.window_s))

        if not world:
            reason = "not_world_frame"
        elif track.age_s(stamp_ns) < p.min_track_age_s or (
                track.hits < p.min_hits):
            reason = "initializing"
        elif history is not None and stamp_ns < history.anomaly_until_ns:
            reason = "shape_jump"
        elif history is not None and self._shape_changing(history):
            reason = "shape_changing"
        elif speed_std > p.max_speed_std:
            reason = "high_uncertainty"
        elif speed > p.max_speed:
            reason = "implausible_speed"
        elif window_v is not None and (
                np.linalg.norm(window_v - v) > p.max_window_mismatch
                or (speed >= p.stationary_speed
                    and np.linalg.norm(window_v) < p.stationary_speed)):
            reason = "inconsistent"
        else:
            reason = None

        if reason is None:
            state = "moving" if speed >= p.stationary_speed else "stationary"
        else:
            state = "unknown"
        return ObjectMotion(
            track.id, state, reason, measured, v, cov, speed, speed_std,
            window_v, track.age_s(stamp_ns), track.since_update_s(stamp_ns),
        )


def assess_boundary(motion, bbox_min, bbox_max, center, boundary,
                    params=None, crane=None, zones=(), world=True):
    """Crane-boundary distance, closing speed and TTC of one object.

    Args:
        motion: ObjectMotion of the object.
        bbox_min, bbox_max: world box of the object.
        center: world center (for exclusion zones).
        boundary: PlaneBoundary or another boundary model.
        params: TtcParams.
        crane: CraneState (static when None).
        zones: ExclusionZones.
        world: False when the box is not in the world frame.
    """
    params = params or TtcParams()
    crane = crane or CraneState()
    if not world:
        return BoundaryRisk(None, None, None, "unknown", None, None,
                            "not_world_frame")

    distance = boundary.distance(bbox_min, bbox_max)
    closing = closing_std = None
    direction = "unknown"
    if motion.valid:
        n = boundary.normal(bbox_min, bbox_max)
        v_rel = motion.velocity - crane.velocity
        closing = float(-(n @ v_rel))
        closing_std = float(np.sqrt(max(n @ motion.velocity_cov @ n, 0.0)))
        if closing > params.min_closing_speed:
            direction = "approaching"
        elif closing < -params.min_closing_speed:
            direction = "receding"
        else:
            direction = "steady"

    def risk(status, ttc=None, ttc_range=None):
        return BoundaryRisk(distance, closing, closing_std, direction, ttc,
                            ttc_range, status)

    if any(zone.contains(center) for zone in zones):
        return risk("excluded")
    if distance <= 0:
        return risk("inside_boundary")
    remaining = distance - params.safety_margin_m
    if remaining <= 0:
        return risk("within_margin")
    if not motion.valid:
        return risk("velocity_unknown")
    if (motion.state != "moving" or direction != "approaching"
            or closing - params.sigma_k * closing_std <= 0):
        return risk("not_approaching")
    ttc = remaining / closing
    if ttc > params.max_ttc_s:
        return risk("beyond_range")
    low = remaining / (closing + params.sigma_k * closing_std)
    slow = closing - params.sigma_k * closing_std
    high = remaining / slow if slow > 0 else None
    return risk("valid", ttc, (low, high))


def _r(value, digits=3):
    if value is None:
        return None
    if np.ndim(value):
        return np.round(np.asarray(value, dtype=np.float64), digits).tolist()
    return round(float(value), digits)


def motion_to_dict(motion):
    """JSON-ready dict of an ObjectMotion.

    velocity_mps / speed_mps are null unless the state is stationary or
    moving; raw_velocity_mps is the unchecked Kalman value (diagnostics).
    """
    valid = motion.valid
    return {
        "state": motion.state,
        "valid": valid,
        "reason": motion.reason,
        "measured": motion.measured,
        "velocity_mps": _r(motion.velocity) if valid else None,
        "speed_mps": _r(motion.speed) if valid else None,
        "speed_std_mps": _r(motion.speed_std),
        "raw_velocity_mps": _r(motion.velocity),
        "track_age_s": _r(motion.track_age_s),
    }


def risk_to_dict(risk):
    """JSON-ready dict of a BoundaryRisk. ttc_s is null unless valid."""
    return {
        "distance_m": _r(risk.distance),
        "closing_speed_mps": _r(risk.closing_speed),
        "closing_speed_std_mps": _r(risk.closing_speed_std),
        "direction": risk.direction,
        "ttc_s": _r(risk.ttc, 2),
        "ttc_range_s": (None if risk.ttc_range is None else
                        [_r(risk.ttc_range[0], 2),
                         _r(risk.ttc_range[1], 2)]),
        "ttc_status": risk.status,
    }


def model_info(boundary, motion_params, ttc_params, crane=None, zones=(),
               reference_frame="world"):
    """Top-level JSON description of how motion and TTC were computed."""
    crane = crane or CraneState()
    return {
        "motion_model": {
            "velocity": "kalman_constant_velocity",
            "reference_frame": reference_frame,
            "components": "horizontal_xy",
            "stationary_speed_mps": motion_params.stationary_speed,
            "max_speed_std_mps": motion_params.max_speed_std,
        },
        "crane_boundary_model": {
            "boundary": boundary.describe(),
            "reference_frame": reference_frame,
            "crane_motion": ("static_assumed" if crane.static else
                             {"velocity_xy": _r(crane.velocity)}),
            "distance": "nearest_box_point_to_boundary",
            "safety_margin_m": ttc_params.safety_margin_m,
            "max_ttc_s": ttc_params.max_ttc_s,
            "min_closing_speed_mps": ttc_params.min_closing_speed,
            "sigma_k": ttc_params.sigma_k,
            "exclusion_zones": [zone.describe() for zone in zones],
            "ttc_definition": ("(distance - safety_margin) / closing_speed,"
                               " current velocity assumed constant"),
            "note": INDICATOR_NOTE,
        },
    }

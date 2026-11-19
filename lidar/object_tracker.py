"""Multi-object tracking: keeps one ID per physical object across frames.

ROS-free, like object_estimator.py. Input per frame is the objects' world
positions (LidarObject.center_reference) and the frame stamp; output is a
track ID per object. The frame-local LidarObject.id is not changed.

Model:
    State [x, y, z, vx, vy, vz] in the reference (world) frame, constant
    velocity with white-noise acceleration. Prediction uses the real stamp
    difference of consecutive frames, never a fixed period: the bag records
    about 12 Hz of sensor time, plays back slower, and frames can be dropped.
Association:
    Hungarian assignment (scipy.optimize.linear_sum_assignment) on the
    negative log-likelihood d^2 + ln|S|, gated by chi-square on d^2 and by a
    maximum horizontal distance from the predicted position.
Lifecycle:
    A new track is tentative and gets confirmed after min_hits updates. Only
    confirmed tracks give out IDs. A track that is not detected keeps being
    predicted ("coasting") and can take its detection back within
    max_coast_s (tentative: tentative_coast_s); after that it is deleted.
    IDs are never reused.
Resets:
    All tracks are dropped when the stamp goes backwards (bag restart), when
    the gap exceeds max_gap_s, or when the reference frame changes (e.g. TF
    lost: positions are then in the cloud frame and not comparable).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


# chi-square, 3 degrees of freedom, 99 %.
CHI2_3DOF_99 = 11.345

_H = np.hstack([np.eye(3), np.zeros((3, 3))])
_INFEASIBLE = 1e9


@dataclass(frozen=True)
class TrackerParams:
    """Tracker parameters (metres, seconds)."""

    accel_std: float = 2.0         # process noise, white acceleration
    meas_std_xy: float = 0.3       # measured center, horizontal
    meas_std_z: float = 0.5        # measured center, vertical
    init_vel_std: float = 3.0      # velocity uncertainty of a new track
    gate_chi2: float = CHI2_3DOF_99
    max_gate_distance: float = 3.0  # horizontal, from the prediction
    min_hits: int = 3
    max_coast_s: float = 1.5       # confirmed track kept without detection
    tentative_coast_s: float = 0.2
    max_gap_s: float = 2.0         # larger stamp gap resets all tracks

    def __post_init__(self):
        positive = ("accel_std", "meas_std_xy", "meas_std_z",
                    "init_vel_std", "gate_chi2", "max_gate_distance",
                    "max_gap_s")
        for name in positive:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        for name in ("max_coast_s", "tentative_coast_s"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        if int(self.min_hits) != self.min_hits or self.min_hits < 1:
            raise ValueError("min_hits must be an integer >= 1")


class Track:
    """One tracked object. State and covariance in the reference frame."""

    def __init__(self, track_id, position, stamp_ns, params):
        self.id = track_id
        self.x = np.concatenate([np.asarray(position, np.float64),
                                 np.zeros(3)])
        self.P = np.diag(np.concatenate([
            _meas_var(params), np.full(3, params.init_vel_std ** 2),
        ]))
        self.hits = 1
        self.confirmed = params.min_hits <= 1
        self.first_stamp_ns = stamp_ns
        self.last_update_ns = stamp_ns

    @property
    def position(self):
        return self.x[:3].copy()

    @property
    def velocity(self):
        return self.x[3:].copy()

    def predict(self, dt, accel_std):
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        # Discrete white-noise acceleration, per axis.
        q = accel_std ** 2
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * (dt ** 4 / 4) * q
        Q[:3, 3:] = Q[3:, :3] = np.eye(3) * (dt ** 3 / 2) * q
        Q[3:, 3:] = np.eye(3) * (dt ** 2) * q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def innovation(self, measurement, R):
        y = np.asarray(measurement, np.float64) - self.x[:3]
        S = self.P[:3, :3] + R
        return y, S

    def update(self, measurement, R, stamp_ns, min_hits):
        y, S = self.innovation(measurement, R)
        K = self.P @ _H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        # Joseph form keeps P symmetric positive definite.
        I_KH = np.eye(6) - K @ _H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.hits += 1
        self.last_update_ns = stamp_ns
        if self.hits >= min_hits:
            self.confirmed = True


def _meas_var(params):
    return np.array([params.meas_std_xy ** 2] * 2 + [params.meas_std_z ** 2])


class ObjectTracker:
    """Assigns persistent track IDs to per-frame object positions."""

    def __init__(self, params=None):
        self.params = params or TrackerParams()
        self.tracks: Dict[int, Track] = {}
        self.next_id = 0
        self.last_stamp_ns: Optional[int] = None
        self.reference_frame: Optional[str] = None
        self.resets = 0

    def reset(self):
        """Drops all tracks. IDs keep counting up, so none is reused."""
        if self.tracks:
            self.resets += 1
        self.tracks = {}
        self.last_stamp_ns = None

    @property
    def confirmed_tracks(self):
        return [t for t in self.tracks.values() if t.confirmed]

    def update(self, stamp_ns, positions, reference_frame=""):
        """Processes one frame.

        Args:
            stamp_ns: Frame stamp in nanoseconds (sensor time).
            positions: (N, 3) object positions in the reference frame.
            reference_frame: Frame of the positions; a change resets.

        Returns:
            List of N entries: the confirmed track ID of each object, or
            None for objects whose track is not confirmed yet.
        """
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        stamp_ns = int(stamp_ns)
        if reference_frame != self.reference_frame:
            self.reset()
            self.reference_frame = reference_frame

        if self.last_stamp_ns is not None:
            dt = (stamp_ns - self.last_stamp_ns) * 1e-9
            if dt < 0 or dt > self.params.max_gap_s:
                self.reset()
            elif dt > 0:
                for track in self.tracks.values():
                    track.predict(dt, self.params.accel_std)
        self.last_stamp_ns = stamp_ns

        R = np.diag(_meas_var(self.params))
        track_list = list(self.tracks.values())
        pairs = self._associate(track_list, positions, R)

        result: List[Optional[int]] = [None] * len(positions)
        matched_tracks = set()
        for t_index, d_index in pairs:
            track = track_list[t_index]
            track.update(positions[d_index], R, stamp_ns,
                         self.params.min_hits)
            matched_tracks.add(track.id)
            if track.confirmed:
                result[d_index] = track.id

        for track in track_list:
            if track.id in matched_tracks:
                continue
            limit = (self.params.max_coast_s if track.confirmed
                     else self.params.tentative_coast_s)
            if (stamp_ns - track.last_update_ns) * 1e-9 > limit:
                del self.tracks[track.id]

        matched_detections = {d for _, d in pairs}
        for d_index in range(len(positions)):
            if d_index in matched_detections:
                continue
            track = Track(self.next_id, positions[d_index], stamp_ns,
                          self.params)
            self.tracks[track.id] = track
            self.next_id += 1
            if track.confirmed:
                result[d_index] = track.id
        return result

    def _associate(self, tracks, positions, R):
        """(track index, detection index) pairs inside the gates."""
        if not tracks or not len(positions):
            return []
        cost = np.full((len(tracks), len(positions)), _INFEASIBLE)
        for i, track in enumerate(tracks):
            S = track.P[:3, :3] + R
            S_inv = np.linalg.inv(S)
            log_det = np.linalg.slogdet(S)[1]
            y = positions - track.x[:3]
            d2 = np.einsum("ni,ij,nj->n", y, S_inv, y)
            horizontal = np.linalg.norm(y[:, :2], axis=1)
            ok = ((d2 <= self.params.gate_chi2)
                  & (horizontal <= self.params.max_gate_distance))
            cost[i, ok] = d2[ok] + log_det
        rows, cols = linear_sum_assignment(cost)
        return [
            (int(r), int(c)) for r, c in zip(rows, cols)
            if cost[r, c] < _INFEASIBLE
        ]

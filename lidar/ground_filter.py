"""NumPy-based tracking constrained RANSAC ground removal.

Plane convention: height(p) = normal @ p + height, where ``normal`` is the
unit world "up" direction expressed in the sensor frame and ``height`` is
the sensor height above the ground plane. Points with
|height(p)| < distance_threshold are ground.

The reference (prior) plane comes from the sensor pose (world -> sensor TF)
and the world ground height. RANSAC only searches near that plane and only
accepts planes close to it, so elevated flat surfaces (crane girders,
truck beds, platforms) are never taken as ground.

Safety policy: if no valid plane is found, nothing is removed
(passthrough). A previous plane is never applied unless max_hold_frames
is explicitly set above 0.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroundParams:
    """Ground removal parameters (metres, degrees)."""

    distance_threshold: float = 0.08
    candidate_band: float = 0.2
    iterations: int = 50
    # Allowed change from the previous valid plane (or prior) per frame.
    max_angle_step_deg: float = 3.0
    max_height_step: float = 0.15
    # Allowed deviation from the TF prior plane, always enforced.
    max_angle_from_prior_deg: float = 10.0
    max_height_from_prior: float = 1.0
    min_inliers: int = 500
    min_inlier_ratio: float = 0.3
    # Frames a previous plane may be reused after a failed fit. 0 = never.
    max_hold_frames: int = 0

    def __post_init__(self):
        positive = (
            "distance_threshold", "candidate_band", "max_angle_step_deg",
            "max_height_step", "max_angle_from_prior_deg",
            "max_height_from_prior",
        )
        for name in positive:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")
        if self.min_inliers < 3:
            raise ValueError("min_inliers must be >= 3")
        if not 0 <= self.min_inlier_ratio <= 1:
            raise ValueError("min_inlier_ratio must be in [0, 1]")
        if self.max_hold_frames < 0:
            raise ValueError("max_hold_frames must be >= 0")


@dataclass(frozen=True)
class GroundResult:
    """Outcome of one frame. mode: 'fit', 'hold' or 'passthrough'."""

    mode: str
    normal: np.ndarray = None
    height: float = None
    candidates: int = 0
    inliers: int = 0
    reason: str = ""


def prior_from_transform(rotation_xyzw, translation, ground_z=0.0):
    """Prior plane from a world <- sensor transform.

    Args:
        rotation_xyzw: Quaternion of the sensor frame in the world frame.
        translation: Sensor origin in the world frame.
        ground_z: World z of the ground plane.

    Returns:
        (normal, height): world up in sensor coordinates, sensor height
        above the ground plane.

    Raises:
        ValueError: if the transform is not finite or not a rotation.
    """
    q = np.asarray(rotation_xyzw, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64)

    if q.shape != (4,) or t.shape != (3,):
        raise ValueError("Expected quaternion (4,) and translation (3,)")
    if not (np.isfinite(q).all() and np.isfinite(t).all()
            and np.isfinite(ground_z)):
        raise ValueError("Transform must be finite")

    norm = np.linalg.norm(q)
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(f"Quaternion is not normalized (|q|={norm:.6f})")
    x, y, z, w = q / norm

    # Third row of the rotation matrix: world z axis in sensor coordinates.
    normal = np.array([
        2.0 * (x * z - y * w),
        2.0 * (y * z + x * w),
        1.0 - 2.0 * (x * x + y * y),
    ])
    return normal, float(t[2] - ground_z)


def _fit_plane(points):
    """Least-squares plane through points: (unit normal, offset)."""
    center = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    normal = vt[2]
    return normal, -normal @ center


def _within(normal, height, ref_normal, ref_height, max_angle_deg,
            max_height):
    return (
        normal @ ref_normal >= np.cos(np.radians(max_angle_deg))
        and abs(height - ref_height) <= max_height
    )


def fit_ground_plane(points, ref_normal, ref_height, prior_normal,
                     prior_height, params, rng):
    """Constrained RANSAC near a reference plane.

    Returns:
        ((normal, height), candidates, inliers); the plane is None if no
        hypothesis satisfies the step, prior and inlier constraints.
    """
    h_ref = points @ ref_normal + ref_height
    cand = points[np.abs(h_ref) < params.candidate_band]

    if len(cand) < params.min_inliers:
        return None, len(cand), 0

    # Random 3-point hypotheses, oriented to point up.
    idx = rng.integers(0, len(cand), size=(params.iterations, 3))
    a, b, c = cand[idx[:, 0]], cand[idx[:, 1]], cand[idx[:, 2]]
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-9
    normals = normals[valid] / lengths[valid, None]
    heights = -np.einsum("ij,ij->i", normals, a[valid])
    flip = np.where(normals @ ref_normal < 0, -1.0, 1.0)
    normals, heights = normals * flip[:, None], heights * flip

    keep = (
        (normals @ ref_normal
         >= np.cos(np.radians(params.max_angle_step_deg)))
        & (np.abs(heights - ref_height) <= params.max_height_step)
    )
    normals, heights = normals[keep], heights[keep]
    if not len(normals):
        return None, len(cand), 0

    counts = (
        np.abs(cand @ normals.T + heights) < params.distance_threshold
    ).sum(axis=0)
    best = int(counts.argmax())
    inlier = (
        np.abs(cand @ normals[best] + heights[best])
        < params.distance_threshold
    )

    # Refine on inliers, then re-check every constraint.
    normal, height = _fit_plane(cand[inlier])
    if normal @ ref_normal < 0:
        normal, height = -normal, -height
    inliers = int(np.count_nonzero(
        np.abs(cand @ normal + height) < params.distance_threshold
    ))

    accepted = (
        _within(normal, height, ref_normal, ref_height,
                params.max_angle_step_deg, params.max_height_step)
        and _within(normal, height, prior_normal, prior_height,
                    params.max_angle_from_prior_deg,
                    params.max_height_from_prior)
        and inliers >= max(params.min_inliers,
                           params.min_inlier_ratio * len(cand))
    )
    return ((normal, height) if accepted else None), len(cand), inliers


class GroundRemover:
    """Stateful ground remover that tracks the plane across frames."""

    def __init__(self, params=None, seed=0):
        self.params = params or GroundParams()
        self.rng = np.random.default_rng(seed)
        self.plane = None   # last plane fitted from data
        self.hold = 0

    def reset(self):
        self.plane = None
        self.hold = 0

    def process(self, points, prior):
        """Classify ground points.

        Args:
            points: (N, 3) XYZ in the sensor frame.
            prior: (normal, height) from prior_from_transform, or None
                when the sensor pose is unknown.

        Returns:
            (ground_mask, GroundResult); ground_mask is an (N,) bool array.
            In passthrough mode it is all False.
        """
        pts = np.asarray(points)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(
                f"Expected points shape (N, 3), got {pts.shape}"
            )
        pts = pts.astype(np.float64, copy=False)
        none = np.zeros(len(pts), dtype=bool)

        if prior is None:
            self.reset()
            return none, GroundResult("passthrough", reason="no prior")

        prior_normal, prior_height = prior
        finite = np.isfinite(pts).all(axis=1)
        valid_pts = pts[finite]

        # Search from the last fitted plane first (it follows slow drift),
        # then from the prior. Both are checked against the prior bounds.
        refs = [(prior_normal, prior_height)]
        if self.plane is not None:
            refs.insert(0, self.plane)

        candidates = inliers = 0
        for ref_normal, ref_height in refs:
            plane, candidates, inliers = fit_ground_plane(
                valid_pts, ref_normal, ref_height, prior_normal,
                prior_height, self.params, self.rng,
            )
            if plane is not None:
                break

        if plane is not None:
            self.plane, self.hold = plane, 0
            mode = "fit"
        elif (
            self.plane is not None
            and self.hold < self.params.max_hold_frames
            and _within(*self.plane, prior_normal, prior_height,
                        self.params.max_angle_from_prior_deg,
                        self.params.max_height_from_prior)
        ):
            self.hold += 1
            plane, mode = self.plane, "hold"
        else:
            return none, GroundResult(
                "passthrough", candidates=candidates, inliers=inliers,
                reason="no valid plane",
            )

        normal, height = plane
        ground = none.copy()
        ground[finite] = (
            np.abs(valid_pts @ normal + height)
            < self.params.distance_threshold
        )
        return ground, GroundResult(
            mode, normal=normal, height=float(height),
            candidates=candidates, inliers=inliers,
        )

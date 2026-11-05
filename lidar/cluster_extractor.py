"""Object candidate extraction: which points belong to which object.

This module only answers "which points form one object". It knows nothing
about position, size or ROS; object_estimator.py computes those from any
group of point indices.

Any extractor implementing ObjectExtractor can be swapped in or run next to
EuclideanClusterExtractor, e.g. a future YOLO frustum extractor that selects
the points projecting inside each 2D detection and receives the detections
through ``context``.
"""

from dataclasses import dataclass
from typing import List, Protocol

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


class ObjectExtractor(Protocol):
    """Groups points into object candidates."""

    def extract(self, points, context=None) -> List[np.ndarray]:
        """Return one int64 index array into ``points`` per candidate.

        Args:
            points: (N, 3) XYZ array.
            context: Extra per-frame input (e.g. 2D detections). Extractors
                that need none ignore it.
        """
        ...


@dataclass(frozen=True)
class ClusterParams:
    """Euclidean clustering parameters (metres, point counts)."""

    tolerance: float = 0.6
    min_points: int = 5
    max_points: int = 3000

    def __post_init__(self):
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and > 0")
        if self.min_points < 1:
            raise ValueError("min_points must be >= 1")
        if self.max_points < self.min_points:
            raise ValueError("max_points must be >= min_points")


def label_components(points, tolerance):
    """Label connected components of the "closer than tolerance" graph.

    Args:
        points: (N, 3) finite array.
        tolerance: Link distance in metres.

    Returns:
        (N,) int array of component labels.
    """
    count = len(points)
    if count == 0:
        return np.empty(0, dtype=np.int32)

    pairs = cKDTree(points).query_pairs(tolerance, output_type="ndarray")
    graph = coo_matrix(
        (np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])),
        shape=(count, count),
    )
    return connected_components(graph, directed=False)[1]


class EuclideanClusterExtractor:
    """Euclidean clustering: connected components within ``tolerance``."""

    def __init__(self, params=None):
        self.params = params or ClusterParams()

    def extract(self, points, context=None):
        """See ObjectExtractor. Non-finite points belong to no candidate.

        Clusters outside [min_points, max_points] are dropped. Candidates
        are ordered by their lowest point index, so the result is
        deterministic for a given input.
        """
        pts = np.asarray(points)
        if pts.size == 0:
            return []
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(
                f"Expected points shape (N, 3), got {pts.shape}"
            )

        valid = np.flatnonzero(np.isfinite(pts).all(axis=1))
        if not len(valid):
            return []

        labels = label_components(
            pts[valid].astype(np.float64), self.params.tolerance
        )
        counts = np.bincount(labels)
        keep = (
            (counts >= self.params.min_points)
            & (counts <= self.params.max_points)
        )

        # Group point positions by label in one sort instead of one
        # comparison per label. The stable sort keeps indices ascending.
        kept = np.flatnonzero(keep[labels])
        order = kept[np.argsort(labels[kept], kind="stable")]
        boundaries = np.cumsum(counts[keep])[:-1]
        groups = np.split(valid[order], boundaries) if len(order) else []
        groups.sort(key=lambda group: group[0])
        return [group.astype(np.int64) for group in groups]

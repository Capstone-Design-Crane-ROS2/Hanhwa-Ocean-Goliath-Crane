"""ROI(Region of Interest) 필터링 함수 (ROS2 의존성 없음, NumPy만 사용).

좌표계: OS1_A (LiDAR 센서 기준, 단위 m)
ROI 범위는 DEFAULT_ROI 값만 바꾸거나, RoiBounds(...)를 새로 만들어 전달하면 된다.
ROS2 노드(lidar_roi_node.py)에서는 파라미터로 덮어쓸 수 있다.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RoiBounds:
    """x, y, z 각 축의 최소·최대 범위 [m]. 경계값은 ROI에 포함된다(min <= v <= max)."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def __post_init__(self):
        for axis in ('x', 'y', 'z'):
            lo = getattr(self, f'{axis}_min')
            hi = getattr(self, f'{axis}_max')
            if not (np.isfinite(lo) and np.isfinite(hi)):
                raise ValueError(f'{axis} 범위는 유한한 값이어야 합니다: ({lo}, {hi})')
            if lo > hi:
                raise ValueError(f'{axis}_min({lo})이 {axis}_max({hi})보다 큽니다.')

    def as_arrays(self):
        """(min_xyz, max_xyz) 형태의 길이 3 배열 두 개 반환."""
        lo = np.array([self.x_min, self.y_min, self.z_min], dtype=np.float64)
        hi = np.array([self.x_max, self.y_max, self.z_max], dtype=np.float64)
        return lo, hi


# 임시 테스트용 ROI (OS1_A 기준). 실제 크레인 주변 ROI 확정 후 수정할 것.
DEFAULT_ROI = RoiBounds(
    x_min=-50.0, x_max=50.0,
    y_min=-30.0, y_max=30.0,
    z_min=-5.0, z_max=15.0,
)


@dataclass(frozen=True)
class RoiStats:
    """필터링 전후 포인트 개수."""

    input_count: int      # 입력 포인트 수
    invalid_count: int    # NaN/Inf 로 제거된 수
    outside_count: int    # ROI 밖이라 제거된 수
    output_count: int     # 최종 남은 수


def filter_roi(points, bounds=DEFAULT_ROI):
    """ROI 필터링.

    Args:
        points: (N, 3) x, y, z 배열. N=0 인 빈 배열도 허용.
        bounds: RoiBounds. 기본값 DEFAULT_ROI.

    Returns:
        (filtered, stats)
        filtered: (M, 3) 배열 (입력 dtype 유지, M <= N)
        stats: RoiStats
    """
    pts = _as_points(points)
    lo, hi = bounds.as_arrays()

    finite = np.isfinite(pts).all(axis=1)
    inside = ((pts >= lo) & (pts <= hi)).all(axis=1)
    keep = finite & inside

    filtered = pts[keep]
    n_in = pts.shape[0]
    n_invalid = int(n_in - np.count_nonzero(finite))
    n_out = int(filtered.shape[0])
    stats = RoiStats(
        input_count=n_in,
        invalid_count=n_invalid,
        outside_count=n_in - n_invalid - n_out,
        output_count=n_out,
    )
    return filtered, stats


def _as_points(points):
    """입력을 (N, 3) 실수 배열로 변환/검증."""
    pts = np.asarray(points)
    if pts.size == 0:
        dtype = pts.dtype if np.issubdtype(pts.dtype, np.floating) else np.float32
        return np.empty((0, 3), dtype=dtype)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f'points 는 (N, 3) 형태여야 합니다. 입력 shape={pts.shape}')
    if not np.issubdtype(pts.dtype, np.floating):
        pts = pts.astype(np.float64)
    return pts

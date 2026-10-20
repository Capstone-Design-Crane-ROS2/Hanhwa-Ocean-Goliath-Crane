"""/point_cloud_360 을 구독하여 ROI 내부 포인트만 /lidar/points_roi 로 발행하는 노드.

실행:
    python3 lidar/lidar_roi_node.py
    python3 lidar/lidar_roi_node.py --ros-args -p roi.x_min:=-20.0 -p roi.x_max:=20.0
"""

import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.node import Node
from rclpy.qos import (
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2

from roi_filter import DEFAULT_ROI, RoiBounds, filter_roi

ROI_KEYS = ('x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max')

XYZ_FIELDS = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]


def cloud_to_xyz(msg):
    """PointCloud2 -> (N, 3) float32 배열. NaN 은 여기서 제거하지 않고 ROI 필터에서 처리."""
    if msg.width * msg.height == 0:
        return np.empty((0, 3), dtype=np.float32)

    field_names = {f.name for f in msg.fields}
    missing = {'x', 'y', 'z'} - field_names
    if missing:
        raise ValueError(f'PointCloud2 에 필드가 없습니다: {sorted(missing)}')

    points = point_cloud2.read_points(
        msg, field_names=('x', 'y', 'z'), skip_nans=False
    )
    # Humble: 구조화 numpy 배열 반환
    if isinstance(points, np.ndarray) and points.dtype.names:
        return np.column_stack(
            (points['x'], points['y'], points['z'])
        ).astype(np.float32, copy=False)
    # 구버전(generator) 대비
    return np.array(list(points), dtype=np.float32).reshape(-1, 3)


def xyz_to_cloud(header, points):
    """(N, 3) 배열 -> xyz float32 PointCloud2. header(stamp, frame_id)는 그대로 사용."""
    pts = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = pts.shape[0]
    msg.fields = XYZ_FIELDS
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True  # ROI 필터에서 NaN/Inf 제거됨
    msg.data = pts.tobytes()
    return msg


class LidarRoiNode(Node):
    def __init__(self):
        super().__init__('lidar_roi_node')

        input_topic = self.declare_parameter('input_topic', '/point_cloud_360').value
        output_topic = self.declare_parameter('output_topic', '/lidar/points_roi').value

        # ROI 범위 [m], OS1_A 기준. 런타임에 ros2 param set 으로도 변경 가능.
        for key in ROI_KEYS:
            self.declare_parameter(
                f'roi.{key}',
                float(getattr(DEFAULT_ROI, key)),
                ParameterDescriptor(description=f'ROI {key} [m] (OS1_A frame)'),
            )
        self.bounds = self._read_bounds()
        self.add_on_set_parameters_callback(self._on_set_parameters)

        # 구독: 센서 데이터용 QoS (BEST_EFFORT) -> RELIABLE/BEST_EFFORT 발행자 모두 수신 가능
        self.subscription = self.create_subscription(
            PointCloud2, input_topic, self.callback, qos_profile_sensor_data
        )
        # 발행: RELIABLE -> RViz2 기본 설정 및 BEST_EFFORT 구독자 모두 호환
        pub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.publisher = self.create_publisher(PointCloud2, output_topic, pub_qos)

        self.get_logger().info(
            f'{input_topic} -> {output_topic}, ROI={self._bounds_str(self.bounds)}'
        )

    def callback(self, msg):
        try:
            points = cloud_to_xyz(msg)
            filtered, stats = filter_roi(points, self.bounds)
        except Exception as e:  # 잘못된 메시지 하나로 노드가 죽지 않도록
            self.get_logger().error(f'포인트클라우드 처리 실패: {e}')
            return

        # 빈 결과도 발행하여 후속 노드가 프레임 누락 없이 timestamp 를 받도록 함
        self.publisher.publish(xyz_to_cloud(msg.header, filtered))

        if stats.input_count == 0:
            self.get_logger().warn('빈 포인트클라우드 수신', throttle_duration_sec=5.0)
            return
        if stats.output_count == 0:
            self.get_logger().warn(
                'ROI 내부 포인트가 없습니다. ROI 범위/frame_id 확인 필요',
                throttle_duration_sec=5.0,
            )
        self.get_logger().info(
            f'stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d} '
            f'frame={msg.header.frame_id} '
            f'in={stats.input_count} out={stats.output_count} '
            f'(invalid={stats.invalid_count}, outside={stats.outside_count})'
        )

    def _read_bounds(self, overrides=None):
        values = {k: float(self.get_parameter(f'roi.{k}').value) for k in ROI_KEYS}
        values.update(overrides or {})
        return RoiBounds(**values)

    def _on_set_parameters(self, params):
        overrides = {}
        for p in params:
            if p.name.startswith('roi.'):
                key = p.name[len('roi.'):]
                if key not in ROI_KEYS:
                    continue
                try:
                    overrides[key] = float(p.value)
                except (TypeError, ValueError):
                    return SetParametersResult(successful=False, reason=f'{p.name}: 숫자가 아닙니다')
            elif p.name in ('input_topic', 'output_topic'):
                return SetParametersResult(
                    successful=False, reason=f'{p.name} 는 실행 시에만 설정할 수 있습니다'
                )
        if not overrides:
            return SetParametersResult(successful=True)
        try:
            new_bounds = self._read_bounds(overrides)
        except ValueError as e:
            return SetParametersResult(successful=False, reason=str(e))
        self.bounds = new_bounds
        self.get_logger().info(f'ROI 변경: {self._bounds_str(new_bounds)}')
        return SetParametersResult(successful=True)

    @staticmethod
    def _bounds_str(b):
        return (
            f'x[{b.x_min}, {b.x_max}] y[{b.y_min}, {b.y_max}] z[{b.z_min}, {b.z_max}]'
        )


def main():
    rclpy.init()
    node = LidarRoiNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

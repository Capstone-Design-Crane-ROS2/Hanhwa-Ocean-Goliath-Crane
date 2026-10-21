import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


class LidarMerger(Node):
    def __init__(self):
        super().__init__('lidar_merger')

        self.subscription = self.create_subscription(
            PointCloud2,
            '/point_cloud_A',
            self.callback,
            10
        )

        self.publisher = self.create_publisher(
            PointCloud2,
            '/point_cloud_360',
            10
        )

        self.buffer = []
        self.last_stamp_ns = None
        self.max_interval_ns = 100_000_000  # 0.1초 (임시 기준)

    def callback(self, msg):
        stamp_ns = (
            msg.header.stamp.sec * 1_000_000_000
            + msg.header.stamp.nanosec
        )

        # 기존 메시지와 시간 및 좌표계 비교
        if self.last_stamp_ns is not None:
            interval_ns = stamp_ns - self.last_stamp_ns

            if interval_ns <= 0:
                self.get_logger().warn(
                    'Timestamp 역행 또는 중복 감지: 버퍼 초기화'
                )
                self.buffer.clear()

            elif interval_ns > self.max_interval_ns:
                self.get_logger().warn(
                    f'메시지 간격 초과 ({interval_ns / 1e9:.3f}s): '
                    '버퍼 초기화'
                )
                self.buffer.clear()

        if self.buffer and (
            msg.header.frame_id != self.buffer[0].header.frame_id
        ):
            self.get_logger().warn(
                'frame_id 불일치: 버퍼 초기화'
            )
            self.buffer.clear()

        self.last_stamp_ns = stamp_ns
        self.buffer.append(msg)

        print(
            f'received frame {len(self.buffer)}/3 '
            f'stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}'
        )

        if len(self.buffer) < 3:
            return

        merged_points = []

        for cloud in self.buffer:
            points = point_cloud2.read_points(
                cloud,
                field_names=('x', 'y', 'z'),
                skip_nans=True
            )

            for p in points:
                x = float(p[0])
                y = float(p[1])
                z = float(p[2])

                if x == 0.0 and y == 0.0 and z == 0.0:
                    continue

                merged_points.append((x, y, z))

        header = Header()
        header.stamp = self.buffer[-1].header.stamp
        header.frame_id = self.buffer[-1].header.frame_id

        merged_cloud = point_cloud2.create_cloud_xyz32(
            header,
            merged_points
        )

        self.publisher.publish(merged_cloud)

        print(
            f'published /point_cloud_360 '
            f'with {len(merged_points)} points'
        )
        print('---')

        self.buffer.clear()


def main():
    rclpy.init()
    node = LidarMerger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

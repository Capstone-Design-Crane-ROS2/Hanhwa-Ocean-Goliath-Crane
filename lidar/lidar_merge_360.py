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

    def callback(self, msg):
        # 이번 120도 메시지를 버퍼에 저장
        self.buffer.append(msg)

        print(
            f'received frame {len(self.buffer)}/3 '
            f'stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}'
        )

        # 3개 모일 때까지 기다림
        if len(self.buffer) < 3:
            return

        merged_points = []

        # 3개 PointCloud2의 x, y, z를 모두 이어붙임
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

                # 무효 포인트 제거
                if x == 0.0 and y == 0.0 and z == 0.0:
                    continue

                merged_points.append((x, y, z))

        # 출력 메시지 header
        header = Header()
        header.stamp = self.buffer[-1].header.stamp
        header.frame_id = self.buffer[-1].header.frame_id

        # xyz PointCloud2 생성
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

        # 다음 3프레임을 위해 초기화
        self.buffer.clear()


def main():
    rclpy.init()
    node = LidarMerger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

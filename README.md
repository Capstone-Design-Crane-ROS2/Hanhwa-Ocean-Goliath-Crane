# Hanhwa-Ocean-Goliath-Crane

ROS2 Humble 기반 골리앗 크레인 상황인식 캡스톤 프로젝트.

## 프로젝트 구조

```
Hanhwa-Ocean-Goliath-Crane/
├── lidar/
│   ├── lidar_merge_360.py   # 120° 스캔 3개 → 360° 결합
│   ├── roi_filter.py        # ROI 필터링 함수
│   └── lidar_roi_node.py    # ROI 필터링 ROS2 노드
├── tests/
│   └── test_roi_filter.py
├── README.md
└── .gitignore
```

## LiDAR

### 토픽

| 노드 | 입력 | 출력 |
|---|---|---|
| `lidar_merge_360.py` | `/point_cloud_A` | `/point_cloud_360` |
| `lidar_roi_node.py` | `/point_cloud_360` | `/lidar/points_roi` |

- 메시지 타입: `sensor_msgs/msg/PointCloud2`, 좌표계: `OS1_A`

### 실행

```bash
source /opt/ros/humble/setup.bash
python3 lidar/lidar_merge_360.py
python3 lidar/lidar_roi_node.py
```

### ROI 범위 (임시값, 단위 m)

X: -50 ~ 50, Y: -30 ~ 30, Z: -5 ~ 15

- 기본값: `lidar/roi_filter.py`의 `DEFAULT_ROI`
- 실행 시 변경: `--ros-args -p roi.x_min:=-20.0 -p roi.x_max:=20.0`

### 테스트

```bash
python3 -m unittest discover -s tests -v
```

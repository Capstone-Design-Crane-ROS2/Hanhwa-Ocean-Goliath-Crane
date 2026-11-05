# LiDAR 객체 추정 모듈 사용법 및 인터페이스

전처리된 포인트클라우드에서 객체 후보를 분리하고 위치·거리·크기·3D 박스를 계산한다.

## 구조

| 파일 | 역할 | 의존성 |
|---|---|---|
| `lidar/cluster_extractor.py` | 어떤 포인트가 한 객체인가 (`ObjectExtractor`, `EuclideanClusterExtractor`) | NumPy, SciPy |
| `lidar/object_estimator.py` | 포인트 그룹 → 위치·거리·크기·박스 (`estimate_objects`) | NumPy |
| `lidar/object_messages.py` | 결과 → JSON, MarkerArray, 라벨 PointCloud2 | ROS 메시지 |
| `lidar/lidar_object_node.py` | 위 모듈을 연결하는 ROS2 노드 | rclpy, tf2_ros |
| `tools/eval_objects_gt.py` | bag과 GT를 직접 읽는 오프라인 정확도 평가 | rclpy 직렬화 |
| `tools/object_topic_monitor.py` | 재생 중 출력 토픽의 누락·중복·정합성 점검 | rclpy |

## 실행

`lidar_ground_node`의 기본 출력은 `/lidar/points_nonground`이므로, 입력 토픽에 맞춰 `output_topic`을 지정한다.

```bash
source /opt/ros/humble/setup.bash
python3 lidar/lidar_merge_360.py
python3 lidar/lidar_roi_node.py
python3 lidar/lidar_voxel_node.py
python3 lidar/lidar_ground_node.py --ros-args -p output_topic:=/lidar/points_preprocessed
python3 lidar/lidar_object_node.py
```

### RViz2와 함께 재생

```bash
rviz2 -d config/rviz/lidar_objects.rviz --ros-args -p use_sim_time:=true
ros2 bag play <bag 디렉터리>/bag
```

**주의**
- RViz2를 `use_sim_time:=true`로 띄울 때는 `--clock`을 쓰지 않는다. bag에 녹화된 `/clock`과 `--clock`의 시계가 함께 발행되어 시간이 되감기고, RViz2가 계속 초기화되어 포인트와 박스가 표시되지 않는다.
- 이 환경에서는 `ROS_DOMAIN_ID=11`만 프로세스 간 통신이 확인되었다(77, 42는 수신 0건). 다른 도메인을 쓰면 오류 없이 수신 0건이 된다.
- 프레임 수를 측정할 때는 bag을 먼저 읽어 둔다(`cat <bag>/bag_0.db3 > /dev/null`). 콜드 읽기에서는 재생이 밀려 프레임이 크게 줄어든다.

### GT 평가

```bash
python3 tools/eval_objects_gt.py <시나리오 또는 bag 디렉터리>... --json result.json
```

## 토픽

입력: `/lidar/points_preprocessed` (`sensor_msgs/PointCloud2`), TF `world -> OS1_A`

| 출력 토픽 | 타입 | 내용 |
|---|---|---|
| `/lidar/objects` | `std_msgs/String` (JSON) | 프레임의 객체 목록 |
| `/lidar/objects/markers` | `visualization_msgs/MarkerArray` | 박스(`object_box`), `#ID 거리` 라벨(`object_label`) |
| `/lidar/objects/points` | `sensor_msgs/PointCloud2` | 객체 포인트, `x y z`(float32) + `object_id`(uint32) |

세 출력은 입력과 같은 `header`(stamp, frame_id)를 사용하며, 입력 프레임마다 객체가 없어도 발행한다.

## 파라미터

`input_topic`, `output_topic`, `marker_topic`, `points_topic`, `world_frame`, `ground_z_world`는 실행 시에만 설정할 수 있다. 나머지는 `ros2 param set`으로 변경할 수 있고, 잘못된 값은 거부된다.

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `cluster.tolerance` | 0.6 | 포인트 연결 거리 [m] |
| `cluster.min_points` | 5 | 최소 포인트 수 |
| `cluster.max_points` | 3000 | 최대 포인트 수 (초과 시 구조물로 간주) |
| `crop.min_height`, `crop.max_height` | -1.0, 10.0 | 지면 기준 높이 범위 [m] |
| `estimate.center_method` | `bbox` | `center` 기준: `bbox` 또는 `centroid` |
| `estimate.max_footprint` | 15.0 | 박스 수평 변 상한 [m], 0이면 해제 |
| `estimate.max_height` | 0.0 | 박스 높이 상한 [m], 0이면 해제 |
| `marker.show_labels` | true | 라벨 표시 여부 |

## `/lidar/objects` JSON (`lidar-objects-v1`)

```json
{
  "schema": "lidar-objects-v1",
  "stamp_ns": 19548366666666,
  "frame_id": "OS1_A",
  "reference_frame": "world",
  "gravity_aligned": true,
  "box_orientation_xyzw": [-0.173648, 0.0, -0.984808, 0.0],
  "object_count": 1,
  "objects": [{
    "id": 0,
    "point_count": 30,
    "center": [0, 0, 0],
    "centroid": [0, 0, 0],
    "bbox_center": [0, 0, 0],
    "distance_m": 0.0,
    "horizontal_distance_m": 0.0,
    "nearest_distance_m": 0.0,
    "size_m": [0, 0, 0],
    "bbox_min_reference": [0, 0, 0],
    "bbox_max_reference": [0, 0, 0],
    "center_reference": [0, 0, 0],
    "corners": [[0, 0, 0]]
  }]
}
```

| 필드 | 좌표계 | 의미 |
|---|---|---|
| `center`, `centroid`, `bbox_center` | `frame_id` | 대표 위치, 포인트 평균, 박스 중심 |
| `corners` | `frame_id` | 8개 꼭짓점: 아래 4개(위에서 볼 때 반시계), 위 4개 |
| `distance_m` | - | 센서에서 `center`까지 3D 거리 |
| `horizontal_distance_m` | - | 위와 같으나 높이 성분 제외 |
| `nearest_distance_m` | - | 객체에서 센서에 가장 가까운 포인트까지의 거리 |
| `size_m` | `reference_frame` 축 | 박스 x, y, z(높이) 길이 |
| `bbox_*_reference`, `center_reference` | `reference_frame` | world 기준 박스와 위치 |
| `box_orientation_xyzw` | - | `frame_id`에서 본 박스 축의 방향 |
| `gravity_aligned` | - | false면 TF 없이 센서 좌표계 기준으로 계산된 값 |

- `id`는 **프레임 내 식별자**로, 거리 오름차순으로 부여한다. 프레임 간 추적이 아니므로 같은 객체의 ID가 바뀔 수 있다.
- 크레인 자체 구조물도 후보로 나온다(센서에서 약 2.7 m, S2·S7 표본 프레임에서 `id` 0). `id` 0을 가장 가까운 실제 객체로 간주하면 안 된다.

## YOLO 융합 담당자용 인터페이스

### 카메라 프레임 대응

`header.stamp`는 360° 합성에 쓰인 **마지막 스캔(t+P)의 시각**이며 카메라 시각이 아니다. `/lidar/merge_sync`로 변환한다.

```python
data = json.loads(msg.data)                       # /lidar/merge_sync
camera_by_lidar[data["lidar_stamps_ns"][-1]] = data["camera_stamp_ns"]
```

### 박스 투영

`corners`(OS1_A) → TF로 `camera_A` → `p_opt = (x, -y, -z)` (데이터셋 카메라 축이 ROS 표준과 다름) → `/camera_info`의 K 적용. 투영 코드는 아직 구현·검증되지 않았다.

### 객체 포인트

`/lidar/objects/points`에서 `object_id`로 선택한다. JSON의 `id`와 같은 값이다.

### 추출 방식 교체·병행

추출과 추정이 분리되어 있어 포인트 그룹만 만들면 나머지는 재사용된다.

```python
class YoloFrustumExtractor:          # ObjectExtractor 구현 예정
    def extract(self, points, context=None):
        # context: 2D 박스 목록. 박스별 포인트 인덱스 배열을 반환한다.
        ...

groups = extractor.extract(points, context=detections)
objects, rejected = estimate_objects(points, groups, pose, params)
```

`LidarObject.point_indices`는 `estimate_objects`에 전달한 배열 기준이다. 노드에서는 높이 범위로 잘린 뒤의 배열이므로, 노드 외부에서는 `/lidar/objects/points`를 사용한다.

# Hanhwa-Ocean-Goliath-Crane

ROS2 Humble 기반 골리앗 크레인 상황인식 캡스톤 프로젝트.

## 프로젝트 구조

```
Hanhwa-Ocean-Goliath-Crane/
├── lidar/
│   ├── lidar_merge_360.py   # 120° 스캔 3개 → 360° 결합 (기존 코드, NUC 테스트 완료)
│   ├── roi_filter.py        # ROI 필터링 함수 (NumPy, ROS2 의존성 없음)
│   └── lidar_roi_node.py    # ROI 필터링 ROS2 노드
├── tests/
│   └── test_roi_filter.py   # ROI 함수 단위 테스트 (ROS2 불필요)
├── README.md
└── .gitignore
```

## LiDAR 전처리 파이프라인

```
/point_cloud_A ──▶ lidar_merge_360.py ──▶ /point_cloud_360 ──▶ lidar_roi_node.py ──▶ /lidar/points_roi ──▶ (지면 제거 / Downsampling)
 (120° ×3)                                 (360°, OS1_A)                              (ROI 내부만, OS1_A)
```

| 노드 | 입력 토픽 | 출력 토픽 | 타입 |
|---|---|---|---|
| `lidar_merger` (`lidar_merge_360.py`) | `/point_cloud_A` | `/point_cloud_360` | `sensor_msgs/msg/PointCloud2` |
| `lidar_roi_node` (`lidar_roi_node.py`) | `/point_cloud_360` | `/lidar/points_roi` | `sensor_msgs/msg/PointCloud2` |

## ROI 필터 (`roi_filter.py`, `lidar_roi_node.py`)

### 역할
- `/point_cloud_360` 에서 x, y, z 추출 → `(N, 3)` NumPy 배열
- NaN / Inf 좌표 제거
- 지정한 x, y, z 최소·최대 범위 밖의 포인트 제거 (**경계값 포함**: `min <= v <= max`)
- 결과를 `/lidar/points_roi` 로 발행, 필터링 전후 포인트 수 로그 출력
- 원본 `header.stamp`, `header.frame_id` 그대로 유지
- 반복문 없이 NumPy 마스크 연산으로 처리

### 출력 메시지 형식 (`/lidar/points_roi`)
- 필드: `x`, `y`, `z` (각 `float32`, offset 0/4/8), `point_step = 12`
- `height = 1`, `width = 포인트 수`, `is_dense = true`, little-endian
- `frame_id`: 입력과 동일 (`OS1_A`), `stamp`: 입력과 동일 (결합된 마지막 120° 메시지 기준)
- ROI 안에 포인트가 하나도 없어도 빈 메시지(`width = 0`)를 발행함

### QoS
- 구독: `qos_profile_sensor_data` (BEST_EFFORT, KEEP_LAST 5) → RELIABLE/BEST_EFFORT 발행자 모두와 호환
- 발행: RELIABLE, KEEP_LAST 5 → RViz2 기본 설정 및 BEST_EFFORT 구독자 모두와 호환

### ROI 파라미터 (OS1_A 좌표계, 단위 m)

| 파라미터 | 기본값 | 비고 |
|---|---|---|
| `roi.x_min` / `roi.x_max` | -50.0 / 50.0 | 임시 테스트 값 |
| `roi.y_min` / `roi.y_max` | -30.0 / 30.0 | 임시 테스트 값 |
| `roi.z_min` / `roi.z_max` | -5.0 / 15.0 | 임시 테스트 값 |
| `input_topic` | `/point_cloud_360` | 실행 시에만 변경 가능 |
| `output_topic` | `/lidar/points_roi` | 실행 시에만 변경 가능 |

> 실제 크레인 주변 ROI 는 아직 확정되지 않았습니다.

변경 방법:

1. **기본값 수정**: `lidar/roi_filter.py` 의 `DEFAULT_ROI`
2. **실행 시 지정**:
   ```bash
   python3 lidar/lidar_roi_node.py --ros-args -p roi.x_min:=-20.0 -p roi.x_max:=20.0 -p roi.z_max:=10.0
   ```
3. **실행 중 변경** (min > max 등 잘못된 값은 거부됨):
   ```bash
   ros2 param set /lidar_roi_node roi.z_min -2.0
   ros2 param get /lidar_roi_node roi.z_min
   ```

### Python 에서 함수 단독 사용
```python
from roi_filter import RoiBounds, filter_roi
filtered, stats = filter_roi(points_xyz, RoiBounds(-20, 20, -10, 10, -2, 10))
print(stats.input_count, stats.output_count)
```

## 테스트

### 로컬 단위 테스트 (ROS2 불필요, NumPy 필요)
```bash
pip install numpy
python3 -m unittest discover -s tests -v
```

### NUC 통합 테스트 (Ubuntu, ROS2 Humble)

모든 터미널에서 먼저:
```bash
source /opt/ros/humble/setup.bash
cd ~/Hanhwa-Ocean-Goliath-Crane   # 실제 clone 경로로 변경
```

1. **ROS bag 재생** (터미널 1)
   ```bash
   ros2 bag info <bag_폴더>                # /point_cloud_A 토픽 존재 확인
   ros2 bag play <bag_폴더> --clock
   ```
2. **360° 결합 노드** (터미널 2)
   ```bash
   python3 lidar/lidar_merge_360.py --ros-args -p use_sim_time:=true
   ```
3. **ROI 노드** (터미널 3)
   ```bash
   python3 lidar/lidar_roi_node.py --ros-args -p use_sim_time:=true
   ```
4. **토픽 발행 확인** (터미널 4)
   ```bash
   ros2 topic list | grep -E "point_cloud_360|points_roi"
   ros2 topic info -v /lidar/points_roi      # QoS, 발행/구독자 확인
   ros2 topic hz /point_cloud_360
   ros2 topic hz /lidar/points_roi           # 두 값이 거의 같아야 함
   ros2 topic echo /lidar/points_roi --field header
   ros2 topic echo /lidar/points_roi --field width
   ```
5. **RViz2 비교**
   ```bash
   rviz2
   ```
   - `Fixed Frame` → `OS1_A`
   - `Add` → `PointCloud2` 2개 추가: `/point_cloud_360`(예: 흰색), `/lidar/points_roi`(예: 빨간색)
   - 표시가 안 되면 해당 Display 의 `Topic > Reliability Policy` 를 `Best Effort` 로 변경
   - 하나씩 체크/해제하며 ROI 밖 포인트가 사라지는지 확인

## 기존 360° 결합 코드(`lidar_merge_360.py`) 개선 검토 사항

> NUC 에서 검증된 코드이므로 **수정하지 않았습니다.** 아래는 추후 팀 논의용 메모입니다.

1. **Timestamp 역행**: bag 을 `--loop` 재생하거나 재시작 시 stamp 가 뒤로 가면 버퍼에 이전 루프 메시지가 섞여 결합될 수 있음. stamp 역행 감지 시 버퍼 초기화 고려.
2. **스캔 경계 정렬**: 단순히 메시지 3개씩 묶으므로, 노드 시작 시점에 따라 0°~360° 가 아닌 다른 120° 조합으로 묶일 수 있음. 메시지 손실 시 이후 묶음이 계속 어긋남.
3. **QoS**: 구독이 RELIABLE(depth 10)이라 LiDAR 드라이버가 BEST_EFFORT 로 발행하면 연결되지 않음. 실센서 연결 시 `qos_profile_sensor_data` 검토.
4. **성능**: 포인트마다 Python 반복문을 사용하므로 포인트 수가 많으면 지연 발생 가능. NumPy 마스크로 대체 가능.
5. **로그**: `print` 대신 `self.get_logger()` 사용 권장.

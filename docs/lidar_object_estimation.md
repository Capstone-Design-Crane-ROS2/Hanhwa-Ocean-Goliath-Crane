# LiDAR 객체 추정 모듈 사용법 및 인터페이스

전처리된 포인트클라우드에서 객체 후보를 분리하고 위치·거리·크기·3D 박스를 계산한다.

## 구조

| 파일 | 역할 | 의존성 |
|---|---|---|
| `lidar/cluster_extractor.py` | 어떤 포인트가 한 객체인가 (`ObjectExtractor`, `EuclideanClusterExtractor`) | NumPy, SciPy |
| `lidar/object_estimator.py` | 포인트 그룹 → 위치·거리·크기·박스 (`estimate_objects`) | NumPy |
| `lidar/object_tracker.py` | 프레임 간 동일 객체 추적, `track_id` 부여 (`ObjectTracker`) | NumPy, SciPy |
| `lidar/object_motion.py` | 칼만 속도의 유효성 판정(`MotionEstimator`), 경계 거리·접근 속도·TTC(`assess_boundary`) | NumPy |
| `lidar/crane_boundary.py` | 크레인 경계 모델(`PlaneBoundary`), 크레인 상태(`CraneState`), 제외 영역 | NumPy |
| `lidar/object_messages.py` | 결과 → JSON, MarkerArray, 라벨 PointCloud2 | ROS 메시지 |
| `lidar/lidar_object_node.py` | 위 모듈을 연결하는 ROS2 노드 | rclpy, tf2_ros |
| `tools/eval_objects_gt.py` | bag과 GT를 직접 읽는 오프라인 정확도 평가 | rclpy 직렬화 |
| `tools/object_topic_monitor.py` | 재생 중 출력 토픽의 누락·중복·정합성 점검 | rclpy |
| `tools/eval_tracking_gt.py` | GT `object_id` 기준 오프라인 추적 평가 (ID Switch, 유지율) | rclpy 직렬화 |
| `tools/eval_tracking_topics.py` | 모니터 기록(실제 발행 JSON)의 `track_id`를 GT와 비교 | rclpy 직렬화 |
| `tools/eval_motion_gt.py` | GT 기준 오프라인 속도·정지 오탐·TTC 상태 평가 (`--cache-dir`로 검출 결과 캐시) | rclpy 직렬화 |
| `tools/eval_motion_topics.py` | 모니터 기록의 `motion`·`crane_boundary` 필드 구조 점검 및 GT 비교 | rclpy 직렬화 |

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
python3 tools/eval_tracking_gt.py <시나리오 또는 bag 디렉터리>... --json tracking.json
#   --drop-rate 0.3 : 프레임 무작위 누락,  --fixed-dt : 고정 주기 가정(비교용)
python3 tools/eval_motion_gt.py <시나리오 디렉터리>... --cache-dir <캐시> --json motion.json
#   --no-validity : 유효성 판정 없이 칼만 속도 그대로(비교용),  --motion-<이름> / --ttc-<이름> : 파라미터
```

재생 중 실제 발행값으로 추적을 확인할 때는 모니터 기록을 GT와 비교한다.

```bash
python3 tools/object_topic_monitor.py --out report.json      # bag 재생 동안 실행
python3 tools/eval_tracking_topics.py report.json <bag 디렉터리>
python3 tools/eval_motion_topics.py report.json <bag 디렉터리>
```

## 토픽

입력: `/lidar/points_preprocessed` (`sensor_msgs/PointCloud2`), TF `world -> OS1_A`

| 출력 토픽 | 타입 | 내용 |
|---|---|---|
| `/lidar/objects` | `std_msgs/String` (JSON) | 프레임의 객체 목록 |
| `/lidar/objects/markers` | `visualization_msgs/MarkerArray` | 박스(`object_box`), `#ID 거리` 라벨(`object_label`), 속도 화살표(`object_velocity`), 속력·TTC 문구(`object_motion`) |
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
| `tracker.enabled` | true | 추적 사용. false면 `track_id` 없이 기존 출력과 동일 |
| `tracker.min_hits` | 3 | 확정에 필요한 연속 매칭 횟수 (확정 전 `track_id`는 null) |
| `tracker.max_coast_s` | 1.5 | 확정 트랙이 미검출 상태로 유지되는 시간 [s]. 이 안에 재검출되면 같은 ID |
| `tracker.tentative_coast_s` | 0.2 | 미확정 트랙 유지 시간 [s] |
| `tracker.max_gate_distance` | 3.0 | 예측 위치와 검출 사이 최대 수평 거리 [m] |
| `tracker.gate_chi2` | 11.345 | 마할라노비스 거리² 게이트 (3자유도 99%) |
| `tracker.accel_std` | 2.0 | 등속 모델 가속도 잡음 [m/s²] |
| `tracker.meas_std_xy`, `tracker.meas_std_z` | 0.3, 0.5 | 중심 측정 잡음 [m] |
| `tracker.init_vel_std` | 3.0 | 새 트랙 속도 불확실성 [m/s] |
| `tracker.max_gap_s` | 2.0 | 프레임 간격이 이보다 크면 모든 트랙 초기화 [s] |

`tracker.*` 값을 실행 중 바꾸면 현재 트랙은 유지된 채 다음 프레임부터 적용된다. `tracker.enabled`를 껐다 켜면 트랙이 초기화된다.

### 속도·TTC 파라미터

기본값은 S1·S2·S4·S7 GT 평가로 고른 **시작값**이며 안전 기준이 아니다. 모두 `ros2 param set`으로 변경할 수 있다.

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `motion.enabled` | true | 속도·TTC 출력. 추적이 꺼져 있으면 무시. false면 추적만 있는 기존 출력과 동일 |
| `motion.min_track_age_s`, `motion.min_hits` | 0.5, 5 | 이보다 어린 트랙의 속도는 `unknown`(`initializing`) |
| `motion.max_speed_std` | 0.6 | 칼만 속도 1σ(수평, 최대 축) 상한 [m/s]. 초과 시 `high_uncertainty` |
| `motion.stationary_speed` | 0.3 | 이 속력 미만이면 `stationary` [m/s] |
| `motion.max_speed` | 15.0 | 물리적으로 불가능한 속력 [m/s] (`implausible_speed`) |
| `motion.size_jump_ratio`, `motion.size_jump_min_m` | 0.3, 0.5 | 박스 수평 변이 한 프레임에 `max(비율×이전, 최소)`보다 변하면 급변 |
| `motion.point_jump_ratio`, `motion.point_jump_min` | 0.5, 10 | 포인트 수 급변 기준 |
| `motion.anomaly_hold_s` | 1.0 | 급변 후 속도를 보류하는 시간 [s] (`shape_jump`) |
| `motion.shape_change_ratio`, `motion.shape_change_min_m` | 0.3, 0.5 | `window_s` 동안 박스 변 길이 변동 폭 기준 (`shape_changing`) |
| `motion.window_s` | 1.0 | 측정 중심 최소제곱 속도(검증용) 창 [s] |
| `motion.max_window_mismatch` | 0.3 | 칼만 속도와 창 속도 차이 상한 [m/s] (`inconsistent`) |
| `ttc.safety_margin_m` | 0.0 | 경계 앞 안전 여유. TTC는 `거리 - 여유`까지의 시간 [m] |
| `ttc.max_ttc_s` | 10.0 | 이보다 큰 TTC는 `beyond_range`, `ttc_s` null [s] |
| `ttc.min_closing_speed` | 0.3 | 접근 판정 최소 접근 속도 [m/s] |
| `ttc.sigma_k` | 1.0 | 접근 판정과 TTC 범위에 쓰는 σ 배수 |
| `crane.face_point_x`, `crane.face_point_y` | 398.884, 0.052 | 크레인 앞면 평면 위의 한 점 (world). **데이터셋 GT `eval_frame` 값**이므로 실제 크레인에 맞게 설정 |
| `crane.face_normal_x`, `crane.face_normal_y` | 1.0, 0.0 | 앞면의 바깥(객체가 오는 쪽) 방향 |
| `crane.velocity_x`, `crane.velocity_y` | 0.0, 0.0 | 크레인 이동 속도(확장용). 현재는 정지 가정 |
| `crane.exclusion_enabled`, `crane.exclusion_boxes` | true, `[388.884, -35, 4.5, 398.884, 35, 12]` | 중심이 들어 있으면 `excluded`인 world 박스 목록(6개 값씩). 기본값은 데이터셋의 크레인 자체 구조물 위치 |
| `marker.show_motion` | true | 이동 객체의 속도 화살표·문구 표시 |
| `marker.velocity_arrow_s` | 1.0 | 화살표 길이 = 속도 × 이 시간 [s] |

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
    "corners": [[0, 0, 0]],
    "track_id": 12
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
| `track_id` | - | 프레임 간 동일 객체 ID. 확정 전이면 null. 추적을 끄면 필드 자체가 없다 |

- `id`는 **프레임 내 식별자**로, 거리 오름차순으로 부여한다. 프레임 간 추적이 아니므로 같은 객체의 ID가 바뀔 수 있다. 프레임 간 동일 객체는 `track_id`로 식별한다.
- `/lidar/objects/points`의 `object_id`와 마커 ID는 계속 `id`(프레임 내 식별자)다. 추적이 켜져 있으면 마커 색과 라벨(`T<track_id>`)만 `track_id`를 따르고, 확정 전 객체는 회색 `#id`로 표시된다.

### 추적 (`track_id`)

- 입력: 객체별 `center_reference`(world)와 프레임 `stamp_ns`. 예측에는 **실제 stamp 차이**를 쓴다(고정 주기 가정 없음). bag은 센서 시간 약 12 Hz가 느리게 재생되고 프레임이 빠질 수 있으므로 수신 간격이 아니라 stamp를 기준으로 한다.
- 모델: 등속 칼만 필터 `[x, y, z, vx, vy, vz]`. 연결: 헝가리안 알고리즘(`scipy.optimize.linear_sum_assignment`), 비용은 `d² + ln|S|`, 마할라노비스 게이트와 최대 수평 거리 게이트를 함께 적용.
- 수명: 새 트랙은 미확정, `min_hits`회 매칭되면 확정되어 `track_id`가 나온다. 검출이 끊겨도 `max_coast_s` 동안 예측 위치로 유지되며, 그 안에 재검출되면 같은 ID를 이어 붙인다. ID는 재사용하지 않는다.
- 초기화: stamp가 뒤로 가거나(bag 재시작), 간격이 `max_gap_s`를 넘거나, `reference_frame`이 바뀌면(TF 상실) 모든 트랙을 버린다.
- 한계: 한 물체가 프레임마다 1개/2개 클러스터로 바뀌면(빈 트럭, 텔레핸들러 붐) 조각마다 트랙이 생겨 GT 기준 ID가 번갈아 바뀐다. 클러스터링 단계의 분리 문제다.
- 크레인 자체 구조물도 후보로 나온다(센서에서 약 2.7 m, S2·S7 표본 프레임에서 `id` 0). `id` 0을 가장 가까운 실제 객체로 간주하면 안 된다.

### 속도와 TTC (`motion`, `crane_boundary`)

> **TTC는 평면 경계 기반 1차 접근 위험 지표다.** 크레인 앞면을 폭 제한 없는 수직 평면으로, 크레인을 정지로 가정하고, 현재 속도가 유지된다고 보고 계산한 예측값이다. 크레인의 실제 외곽·높이·이동은 반영되지 않으므로 충돌 예측이 아니다.

추적이 켜져 있으면 각 객체에 다음 필드가 추가된다(기존 필드는 변경 없음, `schema`도 그대로).

```json
"motion": {
  "state": "moving", "valid": true, "reason": null, "measured": true,
  "velocity_mps": [0.01, 1.98], "speed_mps": 1.98, "speed_std_mps": 0.43,
  "raw_velocity_mps": [0.01, 1.98], "track_age_s": 12.4
},
"crane_boundary": {
  "distance_m": 18.79, "closing_speed_mps": -0.01, "closing_speed_std_mps": 0.42,
  "direction": "steady", "ttc_s": null, "ttc_range_s": null,
  "ttc_status": "not_approaching"
}
```

| 필드 | 의미 |
|---|---|
| `motion.state` | `moving` / `stationary` / `unknown`. `unknown`이면 속도를 쓰지 않는다 |
| `motion.reason` | `unknown`인 이유: `unconfirmed_track`, `not_world_frame`, `initializing`, `shape_jump`, `shape_changing`, `high_uncertainty`, `implausible_speed`, `inconsistent` |
| `motion.velocity_mps`, `speed_mps` | world 수평 속도 `[vx, vy]`와 속력. `unknown`이면 null |
| `motion.speed_std_mps` | 칼만 속도 1σ(최대 축). 칼만 공분산 기반이며 실제 오차보다 큼(S7: σ p50 0.43, 실제 오차 p50 0.05 m/s) |
| `motion.raw_velocity_mps` | 판정 전 칼만 속도(진단용). 확정 전 트랙은 null |
| `motion.measured` | 이번 프레임에 측정된 객체면 true. `predicted_tracks`에서는 false |
| `crane_boundary.distance_m` | 객체 박스의 경계에 가장 가까운 점에서 앞면까지의 부호 있는 수평 거리. 0 이하이면 경계 안. 센서 거리(`distance_m` 등)와 다르다 |
| `crane_boundary.closing_speed_mps` | 경계 쪽 접근 속도 `-n·(v_객체 - v_크레인)`. +면 접근. 속도가 `unknown`이면 null |
| `crane_boundary.direction` | `approaching` / `receding` / `steady` / `unknown` |
| `crane_boundary.ttc_s` | `(distance - safety_margin) / closing_speed`. `ttc_status`가 `valid`일 때만 값이 있고 나머지는 null |
| `crane_boundary.ttc_range_s` | σ를 반영한 `[closing + kσ 기준, closing - kσ 기준]` TTC |
| `crane_boundary.ttc_status` | 아래 표 |

| `ttc_status` | 뜻 |
|---|---|
| `valid` | 이동 중, 접근 속도 > `min_closing_speed` 이고 `closing - kσ > 0`, TTC ≤ `max_ttc_s` |
| `beyond_range` | 접근 중이지만 TTC > `max_ttc_s` |
| `not_approaching` | 정지, 이탈, 평행 이동 또는 접근 속도가 작음 |
| `velocity_unknown` | 속도가 `unknown`이라 보류 |
| `inside_boundary` | 박스가 경계에 닿았거나 넘어섬 (경고와 별개의 상태) |
| `within_margin` | 경계 밖이지만 안전 여유 안 |
| `excluded` | 제외 영역(크레인 자체 구조물) 안 |
| `not_world_frame` | TF가 없어 world 좌표가 아님 |

최상위에 추가되는 필드:
- `motion_model`: 속도 계산 방식, 기준 좌표계, 정지 기준
- `crane_boundary_model`: 경계 모델(`type: plane`, 점, 법선, 폭 무제한), 크레인 이동 가정(`static_assumed`), 거리 정의, 안전 여유, TTC 상한, 제외 영역, TTC 정의, 1차 지표라는 주의 문구
- `predicted_tracks`: 이번 프레임에 검출되지 않고 예측만 된(coasting) 확정 트랙. `track_id`, 예측 `center_reference`, `since_update_s`, `motion`(`measured: false`), `crane_boundary`. 박스는 마지막 측정 크기로 가정한다

마커: 이동(`moving`) 객체에만 속도 화살표(`object_velocity`, 객체 중심에서 `속도 × velocity_arrow_s`)와 `속력 [TTC]` 문구(`object_motion`, 기존 라벨 위)를 표시한다. 유효한 TTC가 있으면 주황-빨강으로 표시한다. 마커 ID는 기존과 같이 `id`다.

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

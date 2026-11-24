# 2차 데이터셋 LiDAR 스캔 전달 (merge `camera_trigger`) 검증 보고

> **요약**: 2차 데이터셋(1메시지 = 360° 스캔 1개, 100 ms)에서 merge 노드가 카메라 프레임 시각에만 스캔을 내보내, bag에 카메라 프레임이 빠진 시각의 스캔이 조용히 버려졌다(S08 285개 중 47개). `camera_trigger:=false`(단일 스캔 전용)를 추가해 모든 스캔을 한 번씩 원래 stamp로 내보내고, `/lidar/merge_sync`는 카메라와 짝이 있는 스캔에만 기존 형식으로 낸다. 오프라인 검증에서 2차 10개 bag의 출력이 bag 스캔과 정확히 일치(누락·중복 0)했고, 1차 35개 bag의 기본 동작은 수정 전과 완전히 같다. ROS2 재생(S08·S09)에서도 `/point_cloud_A` → `/lidar/objects`까지 전 단계 누락·중복 0이다.

작성: 2026-11-24. 브랜치 `feature/lidar-dataset2-validation`. 대상 코드: `lidar/lidar_merge_360.py`.

## 1. 문제와 원인

| 항목 | 내용 |
|---|---|
| 데이터 | 2차 `/point_cloud_A`: 360° 스캔, 100 ms 격자(스캔 stamp mod 100 ms = 0). `/rgb`: 33.3 ms 격자 |
| 기존 동작 | 카메라 프레임마다 `scan_offsets` 위치의 스캔을 찾아 출력(`[0]`이면 카메라 ±2 ms의 스캔 1개). 카메라가 없는 시각의 스캔은 출력 계기가 없어 `evict()`에서 삭제되고, skip 카운트(카메라 기준)에도 나타나지 않는다 |
| 원인 | bag에 카메라 프레임이 빠진 구간이 있다(S08 카메라 간격 66.7 ms 83회, 100 ms 18회). 버려진 47개 스캔의 가장 가까운 카메라는 모두 정확히 ±33.3 ms(+33.3 ms 37개, −33.3 ms 10개): 시각 어긋남이 아니라 그 시각의 카메라 프레임이 없다 |
| 영향 | 버려진 스캔 47개 모두 GT `MEASURING` 구간 |

bag별 스캔(오프라인, bag 기준):

| bag | bag 스캔 | 원본 스캔 공백(복구 불가) | 카메라 정합으로 버려짐 | 기존 출력 |
|---|---|---|---|---|
| S08 | 285 | 6 (200 ms 4곳, 300 ms 1곳) | 47 | 238 |
| S09 | 211 | 0 | 20 | 191 |
| S10 | 261 | 1 (200 ms 1곳) | 22 | 239 |
| S04 | 224 | 0 | 7 | 217 |
| S01 / S02 / S07 | 40 | 0 | 2 / 1 / 1 | 38 / 39 / 39 |
| S03 / S05 / S06 | 40 / 35 / 40 | 0 | 0 | 같음 |

기록 순서: 짝이 있는 경우 카메라가 항상 스캔보다 10–30 ms 늦게 기록된다(S08·S09, 카메라 먼저 0건).

## 2. 변경

### 2.1 `lidar/lidar_merge_360.py`: `camera_trigger` (기본 `true`)

- `true`: 기존 코드 경로 그대로(1차 동작 불변).
- `false`: `scan_offsets:=[0]`일 때만 허용(다른 조합은 시작 시 `ValueError`).
  1. 스캔 도착 즉시 `/point_cloud_360` 발행, stamp·`frame_id`는 원본 그대로(포인트는 기존과 같은 `extract_xyz` → `make_cloud`). 같은 stamp 재수신은 `dup_lidar`로 무시.
  2. 카메라 프레임이 발행된 스캔의 ±`tolerance_ms` 안이면 그 카메라에 대해 기존 형식 `/lidar/merge_sync`(`camera_stamp_ns`, `lidar_stamps_ns: [스캔]`, `point_count`) 1회 발행. 도착 순서 무관(카메라가 먼저면 스캔 도착 시 처리). 스캔 1개당 sync 최대 1개.
  3. 카메라 없는 스캔에는 sync를 만들지 않는다(가짜 stamp 없음).
  4. 진단 카운트: `published`, `sync_published`, `scan_no_camera`(짝 없이 나간 스캔; 버퍼에서 밀려날 때와 종료 시 집계), `camera_no_scan`(스캔 없는 카메라 프레임), 기존 `dup_lidar` / `dup_camera` / `reset`.

### 2.2 설정: `config/lidar/dataset_20261122_lidar.yaml` (신규)

`config/system/dataset_20261122.yaml`(시스템·위험 재생 공용 설정, **변경 안 함**)과 같은 값에 `camera_trigger: false` 한 줄만 추가한 LiDAR 평가 전용 파일. 공용 설정은 `launch/crane_safety.launch.py`, `tools/run_risk_bags.sh`가 사용하므로 위험 재생 결과를 바꾸지 않기 위해 분리했다.

```
$ diff <(sed -n '/^lidar_merge_360:/,$p' config/system/dataset_20261122.yaml) \
       <(sed -n '/^lidar_merge_360:/,$p' config/lidar/dataset_20261122_lidar.yaml)
4a5
>     camera_trigger: false
```

### 2.3 테스트·도구 (신규)

| 파일 | 내용 |
|---|---|
| `tests/test_merge_scan_offsets.py` | 5개 추가: 잘못된 조합 에러, 카메라 없는 스캔 포함 전체 발행·stamp·frame_id·sync, 카메라 선도착 짝짓기, 스캔당 sync 1개·중복 입력 무시, stamp 역행 리셋 |
| `tools/check_merge_offline.py` | 오프라인 비교: 기준 코드(`git show <rev>`) vs 작업 사본, bag 기록 순서대로 콜백에 직접 입력(DDS 없음) |
| `tools/lidar_chain_monitor.py` | 재생 중 `/point_cloud_A` → `/point_cloud_360` → ROI → voxel → ground → `/lidar/objects`, `/lidar/merge_sync` stamp·수신 시각 기록, 누락·중복·순서·지연 보고 |
| `tools/run_lidar_chain_bags.sh` | LiDAR 체인만(merge~object, 위험·카메라·Fusion 없음) launch + 모니터 실행 |

## 3. 결과

### 3.1 단위 테스트

`tests/test_merge_scan_offsets.py` 7개(기존 2 + 신규 5) 성공, `tests/` 전체 240개 성공.

### 3.2 오프라인 (bag 기록 순서, `tools/check_merge_offline.py`, 기준 = 커밋 `37e8605`)

**2차 10개 bag, `camera_trigger:=false`** (synthetic 전 bag + real S08·S09):

| bag | 수정 전 출력 | 수정 후 출력 / bag 스캔 | merge_sync (수정 전과 동일) | scan_no_camera |
|---|---|---|---|---|
| S08 | 238 | **285 / 285** | 238 | 47 |
| S09 | 191 | **211 / 211** | 191 | 20 |
| S10 | 239 | **261 / 261** | 239 | 22 |
| S04 | 217 | **224 / 224** | 217 | 7 |
| S01 / S02 / S07 | 38 / 39 / 39 | **40 / 40** | 38 / 39 / 39 | 2 / 1 / 1 |
| S03 / S05 / S06 | 40 / 35 / 40 | 같음 | 같음 | 0 |

전 bag: 출력 stamp 목록 = 입력 스캔 stamp 목록(정확히 일치), 중복 0, 단조 증가, `frame_id` 유지. merge_sync 카메라–스캔 차 최대 0.0 ms, 카메라·스캔 중복 0, 메시지 목록이 수정 전과 동일. 수정 전에도 나오던 스캔은 포인트 데이터가 바이트 단위로 동일(real: S08·S09). 같은 코드에 기본 `camera_trigger`를 주면 수정 전과 완전히 동일.

**1차 35개 bag 회귀, 기본 설정**: 35/35 수정 전과 완전히 동일(출력 2,485 프레임의 stamp·frame_id·포인트 수·데이터 해시, merge_sync 2,485개, 모든 카운트). real 포인트: S7 트럭(478), S1a(60) 동일.

### 3.3 ROS2 재생 (1배속, LiDAR 체인만, `tools/run_lidar_chain_bags.sh`)

위험·카메라·Fusion 노드는 실행하지 않았다. ROS_DOMAIN_ID 11(실행 전 다른 노드 없음 확인).

| 실행 | 설정 | `/point_cloud_A` | `/point_cloud_360` | ROI / voxel / ground | `/lidar/objects` | 누락 / 중복 / 순서 | merge_sync |
|---|---|---|---|---|---|---|---|
| S08 수정 후 | lidar 전용 | 285 | **285** | 285 / 285 / 285 | **285** | 0 / 0 / 정상 | 228 |
| S09 수정 후 | lidar 전용 | 211 | **211** | 211 / 211 / 211 | **211** | 0 / 0 / 정상 | 185 |
| S09 수정 전 동작 | 공용 (`camera_trigger` 기본) | 211 | 190 | 190 / 190 / 190 | 190 | 21 누락 | 190 |
| 1차 S7 회귀 | 1차 공용 | 1,212 (120°) | **478** (오프라인 기준값과 같음) | 478 / 478 / 478 | **478** | 0 / 0 / 정상 | 478 |

- 재생 중 카메라 메시지 일부가 전송 중 손실되었다(S08 bag 753 → 수신 741, S09 580 → 579). 그래서 merge_sync는 오프라인(238 / 191)보다 적다. 수정 전 동작에서는 이 경우 스캔도 함께 버려지고(S09 190, 오프라인 191), 수정 후에는 스캔은 전부 나가고 sync만 줄어든다.
- 처리시간(노드 자체 집계): merge p50 13.1–13.2 ms, p95 ≤ 16.1 ms; object 노드 total p50 5.3 (S08) / 5.9 ms (S09); 실패 0, TF 없음 0, 트래커 리셋 0.
- `/point_cloud_A` 수신 → `/lidar/objects` 수신 지연(모니터 기준): S09 수정 후 p50 77.5 ms(수정 전 97.9 ms; 카메라 대기 제거로 약 20 ms 감소), p95 157.7 ms. S08 p95 313.8 ms, 최대 371.9 ms: 모니터 자체가 46만 포인트 원본 스캔을 역직렬화하고 bag 플레이어가 "Message queue starved" 경고를 냈기 때문으로 보이나 **원인 미확인**(노드 내부 처리시간은 정상).
- 1차 S7은 수정 전과 같은 코드 경로이며, 프레임 수가 오프라인 기준값과 같다(478, skip 7).

## 4. Fusion·Risk 영향 (실행하지 않음, 코드 근거)

기본값·1차 설정에서는 영향 없음. 2차에서 `camera_trigger:=false`를 쓸 때:

| 대상 | 영향 |
|---|---|
| Fusion `fusion/frame_sync.py` | merge_sync가 같으므로 `full`·`camera_only` 결과 불변 예상. 카메라 없는 스캔의 `/lidar/objects`·points는 sync가 없어 짝 없는 LiDAR로 쌓였다가 60개 초과 시 `orphan_lidar`로 집계·삭제(코드 주석은 "should not happen", 에러 아님) |
| Risk `risk/risk_node.py`, `risk/fusion_link.py` | `/lidar/objects`를 프레임마다 처리하므로 입력 주기가 약 8.9 Hz → 10 Hz(S08). 추가 프레임에는 Fusion 라벨이 없다. 위험 평가 수치(S09 경보 지연, 과경보 프레임)가 달라질 수 있다 |
| `tools/object_topic_monitor.py` | `stamps_with_camera_stamp` < 프레임 수(정상) |
| 공용 설정·launch | 변경 없음. 위험 재생은 계속 `camera_trigger` 기본값(true)으로 동작 |

## 5. 발견한 문제·한계

1. 2차 bag 자체의 스캔 공백(S08 6개, S10 1개)과 카메라 프레임 공백은 데이터 문제로 복구 불가. 트래커는 실제 stamp 간격을 쓰므로 공백을 그대로 처리한다.
2. 재생 중 카메라 메시지 전송 손실(이미지 크기). 수정 후 LiDAR 체인에는 영향이 없다.
3. S08 재생 지연 p95 313 ms의 원인 미확인(모니터/플레이어 부하 추정).
4. 이 NUC에서 ROS_DOMAIN_ID 61은 노드 간 탐색이 되지 않았다(2026-11-24, 같은 도메인의 pub/echo도 실패). 도메인 11만 확인됨.
5. 2차 S08·S09에서 프레임당 객체가 6개다(GT 1개). 배경 정지 클러스터로 보이며 2단계 GT 평가에서 확인한다.

## 6. 재현 방법

```bash
cd ~/Hanhwa-Ocean-Goliath-Crane
D1=~/capstone_data/캡스톤디자인2_ROSbag_20261009
D2=~/capstone_data/캡스톤디자인2_ROSbag_2차_20261122
source /opt/ros/humble/setup.bash

# 단위 테스트
python3 -m pytest -q tests/test_merge_scan_offsets.py

# 오프라인: 기준 커밋 대비 (synthetic 45 bag 약 10분, IO 위주)
python3 tools/check_merge_offline.py /tmp/merge_syn.json synthetic \
    --baseline-rev 37e8605 $D2/S*/*/ $(find $D1 -name metadata.yaml | xargs -n1 dirname | xargs -n1 dirname)
python3 tools/check_merge_offline.py /tmp/merge_real.json real \
    --baseline-rev 37e8605 $D2/S08*/*/ $D2/S09*/*/

# ROS2 재생, LiDAR 체인만 (S08 80 s + S09 56 s + S7 108 s)
tools/run_lidar_chain_bags.sh config/lidar/dataset_20261122_lidar.yaml /tmp/chain_after /point_cloud_A $D2/S08*/*/ $D2/S09*/*/
tools/run_lidar_chain_bags.sh config/system/dataset_20261122.yaml /tmp/chain_before /point_cloud_A $D2/S09*/*/
tools/run_lidar_chain_bags.sh config/system/dataset_20261009.yaml /tmp/chain_ds1 /point_cloud_360 $(ls -d $D1/S7*/*s00*/ | head -1)
```

`check_merge_offline.py`는 경로에 `2차`가 들어간 bag을 단일 스캔 bag으로 다룬다. 기준 커밋 `37e8605`는 `scan_offsets`가 들어간 수정 직전 merge 코드다.

## 7. 다음 단계(2단계) GT 평가 설계

설계만 정리한다. 평가 스크립트는 아직 수정하지 않았다.

### 7.1 확인한 GT 사실 (2026-11-24)

- 2차 `/eval/ground_truth`는 카메라 30 Hz 기준이라 스캔 1개에 MEASURING GT 프레임이 최대 3개(`lidar_dt_s` = −33, 0, +33 ms) 붙는다. 1차는 1,470 프레임 중 스캔 조합 중복 0.
- 2차 프레임별 객체에 `ground_truth_at_lidar` {x, y, z, `stamp_ns`}(eval 좌표)가 있고 `stamp_ns`는 항상 그 프레임의 스캔 stamp와 같다. `lidar_dt_s = 0`이면 `ground_truth`와 같고, 아니면 4–5 cm 다르다.
- 2차 프레임별 `box_corners_world`·`moving`이 있다(카메라 시각 기준). 1차에는 프레임별 박스가 없어 `gt.json`의 박스를 쓴다.
- 프레임별 속도 값은 없다. `gt.json`에 경로점·`speed_mps`·`epoch_sim_s`만 있다.
- S10은 `/tf_static`이 없어 현재 `load_preprocessed`가 그 토픽 조회에서 멈춘다(범위 제외).

### 7.2 수정안

| 문제 | 수정안 | 1차 결과 유지 |
|---|---|---|
| 스캔 1개가 최대 3번 평가됨 | `load_preprocessed`에서 MEASURING GT를 `tuple(lidar_stamps_ns)`로 묶고 스캔마다 1개만 남긴다: `|lidar_dt_s|` 최소(=0) 프레임 우선 | 1차는 중복 0이라 선택이 바뀌지 않는다(같은 프레임 목록) |
| LiDAR 시각 기준 GT | 남긴 프레임의 `lidar_dt_s ≠ 0`이면 박스를 `ground_truth_at_lidar − ground_truth`(eval 좌표 → world: x·dx, y·sy)만큼 평행 이동. `dt = 0`이면 그대로 | 1차에는 필드가 없어 이동 0 |
| 같은 스캔이 트래커에 3번 입력(dt = 0) | 위 중복 제거로 해결. 트래커에는 스캔 stamp(`lidar_stamps_ns[-1]`)를 그대로 준다 | 동일 |
| `eval_*_topics.py`의 GT 매칭 | `frames[lidar_stamps_ns[-1]]` 덮어쓰기를 같은 우선순위(dt = 0 우선)로 바꾼다 | 1차는 키 중복이 없어 불변 |
| S10 | `/tf_static`이 없는 bag은 이유를 출력하고 건너뛴다(평가 범위 제외) | 해당 없음 |

변경은 공용 함수(`load_preprocessed`, `gt_boxes`) 한 곳에 넣고, 1차 35개 bag의 `eval_objects_gt` / `eval_tracking_gt` / `eval_motion_gt` JSON을 수정 전후로 비교해 동일함을 확인한다(수정 전 결과를 먼저 저장).

### 7.3 평가 방식

- **정적 객체(S01–S03, S05–S07)**: 기존 매칭(점 기반 recall ≥ 0.6, purity ≥ 0.5)과 지표(검출률, 상태 ok/split/merged/too_small, 중심 xy 오차 bbox/centroid, 거리 오차, IoU)를 스캔당 1회로 계산. 거리 구간별(8/10/15/20/25/30 m)·클래스별 집계. 배경 클러스터 수(프레임당 객체 − 매칭 객체)도 기록.
- **동적 객체(S04, S08, S09)**: 같은 지표를 LiDAR 시각 GT로 계산. 추적 지표는 기존 정의 그대로: ID Switch(추적 ID가 직전 추적 ID와 다름), fragment(추적 → 미추적 → 추적), MT/ML, 확정 지연(프레임), 공유 ID. ID Switch 시점 전후 프레임의 상태(split/merged/no_points) 변화를 함께 기록해 1차처럼 "클러스터 분리 변화 구간" 여부를 분류한다.
- **속도 GT**: 스캔당 1개로 정리한 LiDAR 시각 GT 중심의 중앙 차분 `(c[k+1] − c[k−1]) / (t[k+1] − t[k−1])`(실제 stamp 차, 스캔 공백 포함). 보조로 `gt.json` 경로(경로점·`speed_mps`)의 해석적 속도와 비교해, 두 값 차가 큰 프레임(출발·정지 직후)을 따로 표시·제외한다. 지표: 속도 크기 오차, 벡터 오차, 방향 오차(|v_gt| ≥ 0.3 m/s), 진행 방향 성분·수직 성분 오차, 트랙 확정 후 경과 시간별 오차(초기 수렴), ID Switch 전후 오차, 유효 판정(`motion.valid`) 비율.
- **입력 경로**: 오프라인 평가 도구는 merge 노드를 거치지 않고 GT의 `lidar_stamps_ns`로 스캔을 직접 고르므로 `camera_trigger`와 무관하게 bag의 모든 스캔(S08 285)을 평가할 수 있다. ROS2 발행값 확인(`eval_*_topics.py`)은 `config/lidar/dataset_20261122_lidar.yaml`로 재생한 결과를 쓴다.

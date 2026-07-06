# ROS 2 기반 자율주행 차량 시뮬레이션 — 고전 주행부터 VLA까지

"ROS 2 기반 자율주행 차량 설계 및 구현" 교재의 주행 시뮬레이션 환경에, **VLA(Vision-Language-Action)**
자율주행 스택을 추가한 워크스페이스입니다.

> 💻 **노트북(GPU)이 없다면 → 브라우저에서 원클릭 실습**
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SKKUAutoLab/VLA_Simulation/blob/main/colab/VLA_Simulation_Colab.ipynb)
> 위 뱃지 클릭 → **런타임 유형을 T4 GPU로 변경 → 런타임 → 모두 실행**. 설치·모델 다운로드·주행이 자동 진행됩니다.
> (사전 학습 산출물은 HuggingFace `hoonsy/VLA_Simulation-pretrained`에서 자동 import → 데이터 수집·학습 생략)

이 README는 두 부분으로 구성됩니다.

- **Part 1 — 시뮬레이터 기본 사용법** : 설치 · 빌드 · 고전(비-VLA) 주행 실행
- **Part 2 — VLA 실습 (교육용)** : *수동 주행 → 데이터 수집 → 학습 → 모델 로드 → 자율주행* 을
  따라 치기만 하면 되도록 정리. VLA의 동작 원리도 함께 설명합니다.

> 대상: 자율주행/딥러닝 입문 학생. 리눅스 터미널과 ROS 2를 처음 다뤄도 순서대로 따라오면 됩니다.

---

# Part 1. 시뮬레이터 기본 사용법

## 1-1. 초기 환경설정

```bash
git clone https://github.com/SKKUAutoLab/ros2_autonomous_vehicle_simulation
cd ~/ros2_autonomous_vehicle_simulation
sh install.sh
source ~/.bashrc
```

```bash
cd ~/ros2_autonomous_vehicle_simulation
export AMENT_PREFIX_PATH=''
export CMAKE_PREFIX_PATH=''
source /opt/ros/humble/setup.bash
rosdep install -i --from-path src --rosdistro humble -y
```

> **VLA 실습(Part 2)을 하려면** 아래 파이썬 패키지가 추가로 필요합니다. `install.sh`에는 포함되어
> 있지 않으므로 별도로 설치하세요(GPU 환경 권장).
> ```bash
> pip install torch transformers peft accelerate
> ```

## 1-2. 패키지 빌드

`interfaces_pkg`(메시지 정의)를 **반드시 먼저** 빌드해야 합니다.

```bash
cd ~/ros2_autonomous_vehicle_simulation
source /opt/ros/humble/setup.bash

colcon build --packages-select interfaces_pkg --allow-overriding interfaces_pkg
source install/local_setup.bash

# 나머지 패키지 (VLA/GUI/미션 패키지 포함)
colcon build --symlink-install --packages-select \
  camera_perception_pkg decision_making_pkg debug_pkg simulation_pkg \
  lidar_perception_pkg mission_control_pkg gui_pkg qwen_vl_pkg
source install/local_setup.bash
```

## 1-3. 고전(비-VLA) 시뮬레이터 실행

```bash
cd ~/ros2_autonomous_vehicle_simulation

# 장애물 없는 기본 주행
sudo killall -9 gazebo gzserver gzclient; ros2 launch simulation_pkg driving_sim.launch.py

# 장애물 + 신호등 미션 주행
sudo killall -9 gazebo gzserver gzclient; ros2 launch simulation_pkg mission_sim.launch.py

# 주차 환경
sudo killall -9 gazebo gzserver gzclient; ros2 launch simulation_pkg parking_sim.launch.py
```

## 1-4. 패키지 & 주요 노드 개요

| 패키지 | 역할 |
|--------|------|
| `interfaces_pkg` | 커스텀 메시지(`MotionCommand`, `LaneInfo`, `Detection` 등) — 모든 패키지의 기반 |
| `camera_perception_pkg` | 카메라 퍼블리시, YOLOv8 검출, 차선 추출, 신호등 검출 |
| `lidar_perception_pkg` | LiDAR 퍼블리시/처리/장애물 검출 |
| `decision_making_pkg` | 경로 계획(`path_planner_node`) + 모션 계획(`motion_planner_node`) |
| `mission_control_pkg` | 미션 태스크 매니저(`task_manager_node`) |
| `simulation_pkg` | Gazebo world/모델 스폰, launch, sim 미러 노드, **`sim_simulation_sender_node`**(제어→cmd_vel 브리지) |
| `debug_pkg` | 시각화(`path_visualizer_node`, `yolov8_visualizer_node`), 이미지 저장 |
| `gui_pkg` | PySide6 GUI, GT 어노테이션 도구(`gt_annotator` 등) |
| `qwen_vl_pkg` | Qwen3-VL 기반 VLA 주행 노드(`vla_brain_node` 등) |

**핵심 제어 메시지** `interfaces_pkg/msg/MotionCommand`
```
int32 steering      # 조향, -7(좌) ~ +7(우)
int32 left_speed    # 좌측 바퀴 속도
int32 right_speed   # 우측 바퀴 속도
```
모든 주행 노드는 이 메시지를 토픽 **`topic_control_signal`** 로 발행하고,
`sim_simulation_sender_node`가 이를 Gazebo의 `/cmd_vel`로 변환합니다.

## 1-5. 고전 주행용 Launch 파일

| Launch 파일 | 용도 |
|-------------|------|
| `driving_sim.launch.py` | 기본 주행 (장애물 없음) |
| `mission_sim.launch.py` | 미션 주행 (장애물 + 신호등) |
| `parking_sim.launch.py` | 주차 |
| `teleop_sim.launch.py` | **수동 주행 최소 구성** (차 + 카메라 + sender, 플래너 없음) — Part 2에서 사용 |

---

# Part 2. VLA 실습 — 수동 주행부터 자율주행까지

> **VLA(Vision-Language-Action)** = "카메라 이미지(Vision)와 언어 명령(Language)을 받아
> 곧바로 주행 행동(Action)을 출력"하는 모델. 이 프로젝트에서는 사람이 직접 몬 주행 데이터를
> 모아 작은 신경망을 학습시키고, 그 모델이 스스로 트랙을 도는 전 과정을 실습합니다.

## 2-0. VLA가 어떻게 동작하는가 (원리)

이 프로젝트의 VLA 주행은 아래 한 줄로 요약됩니다.

```
카메라 이미지 ─▶ [Qwen3-VL 비전 인코더(고정)] ─▶ 70개 토큰 × 2048차원
             ─▶ [작은 학습 헤드(Head)] ─▶ 앞으로 갈 6개 웨이포인트(ex, ey)
             ─▶ [순수추종 Pure-Pursuit] ─▶ MotionCommand(조향/속도)
```

핵심 개념을 학생 눈높이로 정리하면:

1. **비전 인코더는 그대로 두고(freeze), 작은 부분만 학습한다.**
   Qwen3-VL-2B(20억 파라미터)를 처음부터 학습하는 건 불가능하고 불필요합니다.
   대신 비전 인코더는 **고정**하고, 그 위에 얹은 **작은 Head(수 MB)** 만 학습합니다.
   더 성능을 끌어올리고 싶을 때만 **LoRA**(비전 인코더에 얇은 저차원 어댑터를 덧붙여
   극히 일부 파라미터만 학습하는 기법)를 켭니다. → "왜 LoRA인가"의 답.

2. **행동을 '웨이포인트'로 예측한다.**
   조향각을 직접 글자로 뱉게 하는 방식(예: `"D -3 40"`)은 느리고 부정확했습니다.
   그래서 "앞으로 차가 지나갈 6개 점의 좌표(ex=전방, ey=좌측, 단위 m)"를 회귀(regression)로
   예측하고, 고전 제어기(pure-pursuit)가 그 점들을 따라가도록 조향을 계산합니다.

3. **공간 정보를 뭉개지 않는 것이 중요하다.** *(이 프로젝트가 실제로 겪은 교훈)*
   초기에는 70개 토큰을 평균(mean-pool)내어 하나의 벡터로 합쳤는데, 그러면
   "차가 차선 안에서 좌/우 어디에 있는지"라는 **가로 위치 정보가 사라져** 직진 시 좌우로
   흔들렸습니다. 평균 대신 **토큰별 위치 정보를 보존하는 Head(spatial head)**로 바꾸자
   웨이포인트 오차(wpMAE)가 0.56 → 0.284로 개선됐습니다.

4. **역할 분담 — Fast/Slow 이중 시스템.**
   - **Fast (`vla_lora_drive_node`)** : 매 프레임 차선을 보고 실시간으로 웨이포인트→조향 (Part 2 주인공)
   - **Slow (`vla_brain_node`, Qwen3-VL-2B)** : 자유로운 자연어 명령("빨간불에 멈춰")과
     처음 보는 물체를 zero-shot으로 해석해 표준 명령으로 바꿔 `vla/command`로 전달
   - **안전 로직** : 라이다/기하 계산 등 신뢰가 중요한 부분은 LLM이 아니라 결정론적 코드가 담당

> 📁 더 깊은 설계 설명은 `lora_pipeline/SYSTEM_REPORT.md`, `lora_pipeline/DESIGN_pure_vla.md`
> 문서를 참고하세요.

## 2-1. 준비 — 위치와 용어

VLA 학습/주행 스크립트는 별도 디렉터리에 있습니다(ROS 패키지가 아닌 독립 스크립트 모음).

```bash
cd ~/ros2_autonomous_vehicle_simulation/lora_pipeline    # 이하 모든 python3 명령은 워크스페이스 루트에서 실행
```

> ⚠️ **경로 주의:** `lora_pipeline`은 `~/lora_pipeline`이 아니라
> **`~/ros2_autonomous_vehicle_simulation/lora_pipeline`** 에 있습니다.
> 아래 `python3 lora_pipeline/...` 명령은 모두 **워크스페이스 루트(`~/ros2_autonomous_vehicle_simulation`)**
> 에서 실행한다고 가정합니다(스크립트가 자기 위치 기준으로 경로를 계산함).

각 단계는 **별도의 터미널**을 씁니다. 모든 새 터미널에서 먼저 환경을 소스하세요.

```bash
cd ~/ros2_autonomous_vehicle_simulation
source /opt/ros/humble/setup.bash
source install/local_setup.bash
export DISPLAY=:1          # 헤드리스 서버라면 gzclient/창 표시용 (데스크톱이면 생략 가능)
```

---

## 2-2. [1단계] 수동 주행 환경 켜기

**터미널 A** — 차 + 카메라 + 제어 브리지만 있는 최소 시뮬레이터를 띄웁니다(플래너 없음 → 사람이 제어권 독점).

```bash
cd ~/ros2_autonomous_vehicle_simulation
sudo killall -9 gazebo gzserver gzclient 2>/dev/null
ros2 launch simulation_pkg teleop_sim.launch.py
```

## 2-3. [2단계] 키보드로 직접 운전하기

**터미널 B** — 키보드 텔레옵 노드를 실행합니다. 이 터미널에 **포커스**가 있어야 키가 먹습니다.

```bash
cd ~/ros2_autonomous_vehicle_simulation
python3 lora_pipeline/teleop_keyboard.py
# 조향이 반대로 동작하면:  python3 lora_pipeline/teleop_keyboard.py --invert
```

**조작키** (`topic_control_signal`로 `MotionCommand` 발행, 15 Hz)

| 키 | 동작 |
|----|------|
| `w` / `s` | 속도 +20 / −20 (최대 ±255, s로 후진) |
| `a` / `d` | 조향 −1(좌) / +1(우) (최대 ±7) |
| `space` | 정지 (속도 0) |
| `x` | 조향·속도 모두 0으로 리셋 |
| `q` | 종료 |

> 참고: 조향은 자동으로 0으로 돌아오지 않습니다(직접 `a`/`d`로 되돌리거나 `x`로 리셋).
> 스크립트 주석에 나오는 `c`(중앙 정렬) 키는 실제로는 구현돼 있지 않습니다 — `x`를 쓰세요.

트랙을 한두 바퀴 돌며 조작에 익숙해지세요. 차선 중앙을 따라 부드럽게 도는 연습이 좋은 학습 데이터를 만듭니다.

## 2-4. [3단계] 주행 데이터 수집

**터미널 C** — 사람이 운전하는 동안(2-3단계 유지) 이미지 + 조작값을 저장합니다.
`--lane`은 어느 차선을 돌고 있는지 라벨입니다(`0`=1차선/안쪽, `1`=2차선/바깥쪽).

```bash
cd ~/ros2_autonomous_vehicle_simulation
python3 lora_pipeline/manual_collect.py --lane 0
```

- 저장 위치(이미지): `lora_pipeline/manual_demos/images/man_L0_<시간>_<번호>.jpg`
- 저장 위치(라벨): `lora_pipeline/manual_demos/labels.csv`
  - 컬럼: `fname,steering,speed,x,y,yaw,lane`  (이미지파일, 조향, 속도, 위치 x/y, 방향각, 차선)
- 초당 10프레임 저장, **차가 움직일 때만** 기록(정지 프레임은 건너뜀).

> 💡 데이터가 많고 다양할수록(양쪽 차선, 곡선 구간 포함) 모델이 잘 배웁니다.
> `--lane 1`로 바꿔 바깥 차선 데이터도 모으세요.
> `Ctrl+C`로 수집을 마칩니다.

## 2-5. [4단계] 학습용 데이터셋 만들기 (웨이포인트 라벨 생성)

수집한 (이미지 + 위치/방향) 로그를, 모델이 배울 **"앞으로 갈 6개 웨이포인트"** 라벨로 변환합니다.
이 변환은 트랙 중심선 GT 파일이 필요합니다(리포지토리에 이미 포함/홈 디렉터리에 존재).

```bash
cd ~/ros2_autonomous_vehicle_simulation
# 입력: manual_demos/labels.csv  +  ~/track_gt_manual.json, ~/track_gt_outward_centerline.json
python3 lora_pipeline/build_wp_from_manual.py
# 출력: lora_pipeline/manual_wp_labels.csv   (컬럼: path, ex0,ey0 ... ex5,ey5, lane)
```

> 각 프레임에서 차의 현재 위치·방향을 기준으로, 중심선을 따라 앞쪽 6개 지점을
> **차량 좌표계(ex=전방, ey=좌측, m)** 로 투영한 것이 라벨입니다. 즉
> "이 장면에서 사람은 앞으로 이런 궤적으로 갔다"를 모델이 흉내 내도록 가르치는 정답표입니다.

## 2-6. [5단계] 모델 학습

두 가지 방법이 있습니다. **처음에는 (A) 빠른 학습을 권장**합니다.

### (A) 빠른 학습 — 비전 고정 + Head만 학습 *(기본, 권장)*

비전 인코더는 그대로 두고 작은 Head만 학습하므로 빠릅니다(에폭당 수 초~십수 초).

```bash
cd ~/ros2_autonomous_vehicle_simulation
python3 lora_pipeline/train_head_fast.py
# (선택) 에폭 수 조절:  EPOCHS=60 python3 lora_pipeline/train_head_fast.py
# 출력: lora_pipeline/vla_lora_head_fast.pt   ← 주행 노드가 기본으로 로드하는 헤드
```

### (B) 정밀 학습 — 비전 LoRA + Head 동시 학습 *(선택, 더 오래 걸림)*

비전 인코더에 LoRA 어댑터를 달아 함께 학습합니다(에폭 수 24 기본). 수동 데모(`manual_wp_labels.csv`)도 함께 사용됩니다.

```bash
cd ~/ros2_autonomous_vehicle_simulation
python3 lora_pipeline/train_vla_lora.py
# 출력: lora_pipeline/vla_lora_adapter/ (LoRA 어댑터)  +  lora_pipeline/vla_lora_head.pt
```

> **무엇을 배우나?** 두 방법 모두 최종 목표는 "이미지 → 6개 웨이포인트" 매핑입니다.
> (A)는 Head MLP만, (B)는 비전 특징까지 미세조정(LoRA)한다는 차이입니다.
> Head 구조: `토큰별 Linear(2048→64) + 위치 임베딩 + flatten + FiLM(차선 조건) → MLP → 12(=6×2)`.

## 2-7. [6단계] 학습한 모델로 자율주행

**터미널 A의 시뮬레이터는 끄고**, VLA 주행 launch를 실행합니다. 이 launch 하나가
Gazebo · 차량 스폰 · 제어 브리지 · **VLA 주행 노드**를 모두 띄웁니다.

```bash
cd ~/ros2_autonomous_vehicle_simulation
sudo killall -9 gazebo gzserver gzclient 2>/dev/null
source install/setup.bash
ros2 launch lora_pipeline/vla_drive.launch.py
# 선택 인자:  brain:=false (Qwen 브레인 노드 끄기)  gui:=false (GUI 끄기)  gzclient:=true (3D 창 보기)
```

- 주행 노드(`vla_lora_drive_node.py`)는 기본적으로 **비전 고정 + `vla_lora_head_fast.pt`** 를 로드합니다.
- (B)에서 만든 **LoRA 어댑터**로 주행하려면 환경변수로 켜세요:
  ```bash
  VLA_USE_ADAPTER=1 VLA_HEAD=$PWD/lora_pipeline/vla_lora_head.pt \
  VLA_ADAPTER=$PWD/lora_pipeline/vla_lora_adapter \
  ros2 launch lora_pipeline/vla_drive.launch.py
  ```

**주행 노드 토픽**
- 구독: `camera/image_raw`(이미지), `/odom`(위치), `scan`(라이다, 전방 정지/회피), `vla/command`(명령)
- 발행: `topic_control_signal`(`MotionCommand`), `vla/cur_lane`(현재 차선)

## 2-8. [7단계] 자연어 명령 보내기

**새 터미널** — 주행 노드에 명령을 보냅니다(예: 1차선으로 한 바퀴).

```bash
source ~/ros2_autonomous_vehicle_simulation/install/local_setup.bash
ros2 topic pub --once /vla/command std_msgs/String "{data: '1차선 한바퀴 돌아'}"
# 정지:            "{data: '멈춰'}"
# 2차선 주행:      "{data: '2차선 한바퀴 돌아'}"
```

> `vla/command`는 키워드로 해석됩니다(차선/랩수/정지/랜드마크 등). 완전 자유형 문장은
> `vla_brain_node`(Qwen3-VL)가 `/nl_command`로 받아 표준 명령으로 변환해 `vla/command`로 넘깁니다.

## 2-9. [8단계] 평가 (선택)

주행 노드가 도는 상태에서 정량 평가:

```bash
cd ~/ros2_autonomous_vehicle_simulation
# 한 바퀴 폐루프 평가 (중심선 이탈/커버리지/조향 지터)
python3 lora_pipeline/eval_lap.py --lane 0            # --lane 1 로 2차선 평가

# 한 바퀴 자동 완료 판정 (>20m 주행 후 출발점 3m 이내 복귀 & 정지 시 성공)
python3 lora_pipeline/one_lap_test.py
```

---

## 2-10. 전체 흐름 한눈에 보기

```
[1] teleop_sim.launch.py        시뮬레이터(차+카메라+sender)
        │
[2] teleop_keyboard.py          사람이 키보드로 운전  (w/s/a/d/space/x/q)
        │
[3] manual_collect.py --lane 0  이미지 + 조작/위치 저장 → manual_demos/{images, labels.csv}
        │
[4] build_wp_from_manual.py     웨이포인트 라벨 생성   → manual_wp_labels.csv
        │
[5] train_head_fast.py          학습(비전고정+Head)    → vla_lora_head_fast.pt
    (또는 train_vla_lora.py:      LoRA 정밀학습          → vla_lora_adapter/ + vla_lora_head.pt)
        │
[6] vla_drive.launch.py         학습모델로 자율주행     (vla_lora_drive_node)
        │
[7] ros2 topic pub /vla/command 자연어 명령 전달
        │
[8] eval_lap.py / one_lap_test.py   성능 평가
```

---

## 2-11. VLA 주행 노드 참고 (qwen_vl_pkg)

Part 2는 `lora_pipeline`의 경량 주행 노드를 사용하지만, `qwen_vl_pkg`에는 Qwen3-VL을 직접 쓰는
상위 VLA 노드들도 있습니다(`ros2 run qwen_vl_pkg <노드>`).

| 실행자 | 설명 | 관련 launch |
|--------|------|-------------|
| `vla_brain_node` | Qwen3-VL + 기하 내비 + 장애물 감속(상위 브레인) | `vla_brain_sim.launch.py`, `full_mission_sim.launch.py` |
| `vla_agent_node` | 순수 VLA(매 사이클 tool-call 제어), `~/track_features.json` 필요 | `vla_agent_sim.launch.py` |
| `qwen_vl_driver_node` | end-to-end Qwen3-VL-2B 드라이버 | `qwen_vl_sim.launch.py` |
| `vla_cmd_node` | 자연어 목표를 `vla/goal_cmd`로 발행하는 터미널 헬퍼 | — |

```bash
# 예: VLA 브레인 + 미션 통합 (기본 use_vla:=true, use_gui:=true)
ros2 launch simulation_pkg full_mission_sim.launch.py
ros2 launch simulation_pkg full_mission_sim.launch.py use_vla:=false   # 고전 플래너로 대체
```

---

## 2-12. 알아두면 좋은 주의사항 (학생용 트러블슈팅)

1. **경로 혼동:** `lora_pipeline`은 `~/lora_pipeline`이 아니라
   `~/ros2_autonomous_vehicle_simulation/lora_pipeline`. `python3 lora_pipeline/...`는 워크스페이스 루트에서.
2. **`teleop_sim.launch.py`는 키보드 노드를 켜지 않습니다.** `teleop_keyboard.py`를 별도 터미널에서 실행하세요.
3. **`simulation_pkg`의 `data_collection_node`는 소스가 없어 실행되지 않습니다.**
   데이터 수집은 반드시 `lora_pipeline`의 `manual_collect.py`(사람 주행) 또는
   `collect_demos_node.py`(고전 스택을 교사로 삼는 자동 수집)를 사용하세요.
4. **기본 주행 모드는 LoRA 어댑터가 아니라 '비전 고정 + spatial head'입니다.**
   LoRA로 주행하려면 `VLA_USE_ADAPTER=1`을 명시하세요(코드 기준. 일부 주석은 오래되어 다름).
5. **두 종류의 'LoRA'가 공존합니다.** `train_lora.py`/`eval_lora.py`는 "글자 조향(`D st sp`)" 방식(출력 `adapter/`),
   Part 2의 주행 노드가 쓰는 것은 "웨이포인트 헤드" 방식(`vla_*head*.pt`). 서로 다른 파이프라인입니다.
6. **VLA 노드는 `torch`/`transformers`/`peft`가 필요합니다.** `install.sh`에 없으니 Part 1의 안내대로 별도 설치.
7. **`run_sim.sh`는 시뮬레이터(차+sender)만 띄웁니다.** VLA 주행까지 한 번에 하려면 `vla_drive.launch.py`를 쓰세요.

---

# 부록 A. 트랙 차선 GT 시각화

VLA 학습에 쓰이는 **트랙 차선 중심선 Ground-Truth(GT)** 데이터를 눈으로 확인하는 방법입니다.
(Part 2의 [4단계 `build_wp_from_manual.py`]가 이 GT 파일들을 웨이포인트 라벨 생성에 사용합니다.)
**① 좌표 평면 플롯**, **② 트랙 이미지 위 오버레이**, **③ 실제 GUI 창 스크린샷(중심선 / 도로·차선 마스킹)**
세 가지 방식이 있습니다.

## A-1. 관련 파일 위치

### GT 데이터 파일 (`~/`)
| 파일 | 내용 | 좌표 형식 |
|------|------|-----------|
| `track_gt_manual.json` | 수동 어노테이션 (lane=inner, 720점) | `centerline_pixels` + `centerline_world` |
| `track_gt_lane0_demo.json` | 0차선(outer) 데모 (720점) | `centerline_world` |
| `track_gt_lane1_demo.json` | 1차선(inner) 데모 (720점) | `centerline_world` |
| `track_gt_outward_centerline.json` 등 | 기타 차선/중심선 GT | 파일별 상이 |

> **주의:** GUI(`gt_annotator`)는 `centerline_pixels`(픽셀 좌표)를 직접 읽습니다.
> `*_demo.json` 처럼 `centerline_world`(월드 좌표)만 있는 파일은 `world_to_pixel()` 변환이 필요합니다.

### GT를 그리는 원본 코드 (`src/gui_pkg/gui_pkg/`)
| 파일 | 위치 | 역할 |
|------|------|------|
| `gt_annotator.py` | 라인 1250~1360 (`Canvas.paintEvent`) | GUI 캔버스에 도로/차선 마스크·중심선·수동점 렌더링 |
| `gt_annotator.py` | 라인 1323~1350 | 1차선(청록)·2차선(주황)·수동(빨강) 중심선 점 |
| `track_map_widget.py` | 라인 225~240 (`_draw_centerline`) | 중심선을 노란 점선 + 점으로 표시 |
| `gt_extractor.py` | 라인 209~248 (`visualize_extraction`) | OpenCV 도로 마스크 오버레이 + 중심선 점 |

### 좌표 변환식 (`gt_annotator.py`, `IMG_W=1180.0`, `IMG_H=884.0`)
```python
def pixel_to_world(px, py):
    return -20.237 + (py / IMG_H) * 40.473,  -26.915 + (px / IMG_W) * 53.83

def world_to_pixel(wx, wy):
    return int((wy + 26.915) / 53.83 * IMG_W),  int((wx + 20.237) / 40.473 * IMG_H)
```

### 트랙 배경 이미지
```
src/simulation_pkg/models/race_track/materials/textures/track.png   (1181 x 885)
```

## A-2. 실행 방법

시각화 스크립트는 **[`tools/`](tools/)** 에 정리되어 있습니다(사용법: [`tools/README.md`](tools/README.md)).
모든 명령은 워크스페이스 루트에서 실행합니다.

```bash
# (A) 월드좌표 평면 플롯 — matplotlib
python3 tools/gt_render.py ~/track_gt_manual.json -o /tmp/gt_render.png

# (B) 트랙 이미지 위 오버레이 — OpenCV
python3 tools/gt_overlay.py ~/track_gt_manual.json -o /tmp/gt_overlay.png

# (C) 실제 GUI 창 스크린샷 (헤드리스면 DISPLAY=:1 QT_QPA_PLATFORM=xcb 필요)
DISPLAY=:1 QT_QPA_PLATFORM=xcb python3 tools/gt_gui_shot.py ~/track_gt_manual.json -o /tmp/gt_gui.png
#   도로/차선 마스킹(1차선 빨강 / 2차선 파랑)까지 그리려면 --mask 추가:
DISPLAY=:1 QT_QPA_PLATFORM=xcb python3 tools/gt_gui_shot.py ~/track_gt_manual.json --mask -o /tmp/gt_mask.png
```

- 입력 GT는 생략 시 `~/track_gt_manual.json`. `centerline_pixels`가 없으면 `centerline_world`를 자동 변환.
- `gt_gui_shot.py`는 `src/gui_pkg/gui_pkg/gt_annotator.py`를 import하므로 저장소 안에서 실행해야 합니다.

> **함정 노트:** `gt_annotator`의 `set_show_lane()` 인자는 `"both" | "inner" | "outer"` 문자열입니다.
> `True` 같은 값을 넣으면 마스크가 렌더링되지 않습니다 (실제로 겪은 버그).

### (D) 어노테이터 GUI 직접 실행 (ROS 실행자)
```bash
cd ~/ros2_autonomous_vehicle_simulation
source install/local_setup.bash
ros2 run gui_pkg gt_annotator      # 트랙 GT 어노테이터
ros2 run gui_pkg gt_extractor      # 도로 세그멘테이션 기반 중심선 추출
```

## A-3. 도로/차선 마스킹 파이프라인 (참고)

`gt_annotator.py`의 `Worker`(QThread) 처리 순서:
1. **도로 마스크** — `detect_road_by_color()`(색상 피커) 또는 `detect_road_by_v()`(V 범위)
2. **내부 타원 경계 검출** — `find_ring_boundary()` (주차장 영역 제외용)
3. **차선 분리** — `split_lanes_polar()` 극좌표 기반 inner/outer 분리
4. **중심선 추출** — `polar_centerline()` (720 bins + 스무딩)

결과는 캔버스에 1차선 빨강 / 2차선 파랑 반투명 마스크로 오버레이됩니다.

---

## 참고 문서

- `lora_pipeline/SYSTEM_REPORT.md` — Fast/Slow 이중 시스템, spatial head 개선기, 추론 파이프라인 상세
- `lora_pipeline/DESIGN_pure_vla.md` — 순수 단일 VLA 설계 원칙과 마일스톤
- `lora_pipeline/VLA_node_architecture.pptx` — 노드/토픽 블록 다이어그램

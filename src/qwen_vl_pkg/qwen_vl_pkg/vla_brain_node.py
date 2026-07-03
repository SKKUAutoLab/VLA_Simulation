#!/usr/bin/env python3
"""
VLA Brain Node — Goal-conditioned autonomous driving

명령 수신 방법:
    ros2 topic pub /vla/goal_cmd std_msgs/msg/String \
        "data: '신호등까지 가줘'" --once

지원 명령 예시:
    "신호등까지 가줘"         → traffic_light 목표, track 모드
    "신호등 직접 가줘"        → traffic_light 목표, direct 모드
    "출발점으로 돌아가"        → start 목표, direct 모드
    "멈춰"  /  "정지"         → 즉시 정지
    "출발"  /  "계속 가줘"    → track 모드 재개

직접 경로(direct) 모드:
    Gazebo /gazebo/model_states 에서 자차 위치를 실시간으로 읽고
    목표 좌표까지의 기하학적 heading 오차로 조향 제어.

장애물 안전:
    LiDAR(lidar_processed) 전방 거리 → 속도 감쇄
      > 3.0m : 속도 × 1.0 (정상)
      > 2.0m : 속도 × 0.5 (감속)
      > 1.0m : 속도 × 0.2 (徐行)
      ≤ 1.0m : 정지
"""

import math
import threading
import time
import json
import re

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from interfaces_pkg.msg import MotionCommand

import cv2
import numpy as np
from PIL import Image as PILImage

import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from nav_msgs.msg import Odometry

# ─── 모델 상수 ─────────────────────────────────────────────────────────────
MODEL_NAME      = "Qwen/Qwen3-VL-2B-Instruct"
CAMERA_TOPIC    = "camera/image_raw"
CONTROL_TOPIC   = "topic_control_signal"
GOAL_CMD_TOPIC  = "vla/goal_cmd"
LIDAR_TOPIC     = "lidar_processed"   # LiDAR 전처리 후 토픽
ODOM_TOPIC = "/odom"

PUBLISH_HZ      = 10.0
INPUT_W, INPUT_H = 640, 480
PROCESSOR_PIXELS = INPUT_W * INPUT_H
MAX_NEW_TOKENS  = 16

# ─── 사전 정의 웨이포인트 (Gazebo world 좌표) ─────────────────────────────
WAYPOINTS = {
    "start":         (-2.55, -22.71),   # 차량 스폰 구역 중심
    "traffic_light": (-5.63,  17.90),   # 신호등 위치
    "obstacle_1":    (-3.66,   8.71),   # 장애물 구역 1
    "obstacle_2":    (-3.66,   2.04),   # 장애물 구역 2
    "mid":           (-2.55,  -2.00),   # 트랙 중간 지점 (추정)
}

GOAL_REACHED_DIST = 3.0   # 목표 도달 판정 거리 [m]

# ─── 장애물 속도 감쇄 테이블 ──────────────────────────────────────────────
OBSTACLE_SPEED_TABLE = [
    (3.0, 1.0),
    (2.0, 0.5),
    (1.0, 0.2),
    (0.0, 0.0),
]

# ─── 조향 비례 이득 (direct 모드) ─────────────────────────────────────────
HEADING_GAIN = 4.5    # heading error (rad) → steering (-7 ~ 7)

# ─── VLM 프롬프트 (track 모드) ────────────────────────────────────────────
def make_track_prompt(goal_name: str, goal_desc: str) -> str:
    return f"""/no_think
You are an autonomous vehicle. Goal: {goal_desc}.
Follow the road/track toward the goal.
If the goal object is visible and close (large in image), set speed to 0 (goal reached).
Output ONLY this JSON:
{{"steering": <int -7 to 7>, "speed": <int 0 to 100>, "goal_reached": <bool>}}
Steering: -7=hard left, 0=straight, +7=hard right. Speed: 0=stop, 100=full."""


def make_direct_prompt(goal_name: str, heading_error_deg: float, dist: float) -> str:
    direction = "LEFT" if heading_error_deg < -5 else ("RIGHT" if heading_error_deg > 5 else "AHEAD")
    return f"""/no_think
Driving DIRECTLY to '{goal_name}' (ignore lane markings).
Target is {direction} ({heading_error_deg:+.0f}°), {dist:.1f}m away.
Output ONLY this JSON:
{{"steering": <int -7 to 7>, "speed": <int 0 to 100>, "goal_reached": <bool>}}
Adjust steering toward target. Goal reached if distance < 2m."""


# ─── 유틸 ─────────────────────────────────────────────────────────────────
def quat_to_yaw(qx, qy, qz, qw) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def normalize_angle(a: float) -> float:
    while a > math.pi:  a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


def obstacle_speed_factor(min_dist: float) -> float:
    for threshold, factor in OBSTACLE_SPEED_TABLE:
        if min_dist > threshold:
            return factor
    return 0.0


def parse_goal_cmd(text: str) -> dict:
    """
    자연어 명령 → {target, mode, action}
    action: "go" | "stop" | "resume"
    mode:   "track" | "direct"
    target: waypoint key 문자열

    ※ 목표 키워드 감지를 resume 보다 먼저 수행:
      "출발점으로 돌아가" → '출발' 포함이지만 목표가 있으므로 "go"
    """
    t = text.strip().lower()

    # ① 정지 (최우선)
    if any(k in t for k in ["멈춰", "정지", "stop", "멈춤"]):
        return {"action": "stop", "target": None, "mode": None}

    # ② 목표 감지 (resume보다 먼저 → "출발점" 등 오파싱 방지)
    target = None
    if any(k in t for k in ["출발점", "처음", "돌아가", "스폰", "원점"]):
        target = "start"
    elif "start" == t.strip():          # 단독 'start' 키워드
        target = "start"
    elif any(k in t for k in ["신호등", "traffic"]):
        target = "traffic_light"
    elif any(k in t for k in ["장애물1", "obstacle_1", "obstacle1"]):
        target = "obstacle_1"
    elif any(k in t for k in ["장애물2", "obstacle_2", "obstacle2"]):
        target = "obstacle_2"
    elif any(k in t for k in ["중간", "mid", "midpoint"]):
        target = "mid"

    if target is not None:
        # "돌아가" / "복귀" 는 출발점 직접 이동을 의미하므로 direct 기본
        direct_kws = ["직접", "바로", "direct", "무시", "돌아가", "복귀"]
        mode = "direct" if any(k in t for k in direct_kws) else "track"
        return {"action": "go", "target": target, "mode": mode}

    # ③ 재개 (목표 없을 때만)
    if any(k in t for k in ["출발", "계속", "go", "resume", "시작"]):
        return {"action": "resume", "target": None, "mode": None}

    # ④ 기본: traffic_light (track)
    mode = "direct" if any(k in t for k in ["직접", "바로", "direct", "무시"]) else "track"
    return {"action": "go", "target": "traffic_light", "mode": mode}


# ─── 메인 노드 ────────────────────────────────────────────────────────────
class VLABrainNode(Node):
    def __init__(self):
        super().__init__("vla_brain_node")

        # ── 상태 ──
        self.goal = None          # dict: {target, mode}
        self.is_stopped = False   # 사용자 정지 명령
        self.goal_reached = False

        # 자차 위치 (Gazebo) — pose_lock으로 race condition 방지
        self.car_x = None
        self.car_y = None
        self.car_yaw = None
        self.pose_lock = threading.Lock()

        # 장애물
        self.obstacle_min_dist = 999.0

        # VLM
        self.latest_image: Image | None = None
        self.is_inferring = False
        self.infer_lock = threading.Lock()

        # 최종 출력값
        self.steering = 0
        self.speed    = 0
        self._infer_times: list[float] = []

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        # ── 모델 로드 ──
        self.get_logger().info(f"Loading {MODEL_NAME} (sdpa) ...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16,
            device_map="cuda:0", attn_implementation="sdpa",
        )
        self.model.eval()
        self.processor = Qwen3VLProcessor.from_pretrained(
            MODEL_NAME, min_pixels=PROCESSOR_PIXELS, max_pixels=PROCESSOR_PIXELS,
        )
        self._warmup()
        self.get_logger().info(
            f"VLA Brain ready. VRAM: {torch.cuda.memory_allocated()//1024**2} MB"
        )

        # ── 구독 ──
        self.create_subscription(Image, CAMERA_TOPIC, self._image_cb, qos)
        self.create_subscription(String, GOAL_CMD_TOPIC, self._goal_cmd_cb, qos)
        self.create_subscription(LaserScan, LIDAR_TOPIC, self._lidar_cb, qos)

        # /odom 에서 자차 위치 취득
        self.create_subscription(
            Odometry, ODOM_TOPIC, self._odom_cb,
            QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                       history=QoSHistoryPolicy.KEEP_LAST,
                       durability=QoSDurabilityPolicy.VOLATILE)
        )

        self.pub   = self.create_publisher(MotionCommand, CONTROL_TOPIC, qos)
        self.status_pub = self.create_publisher(String, "vla/status", qos)
        self.timer = self.create_timer(1.0 / PUBLISH_HZ, self._publish_cb)

        self.get_logger().info(
            "Waiting for goal command...\n"
            f"  → ros2 topic pub /{GOAL_CMD_TOPIC} std_msgs/msg/String "
            "\"data: '신호등까지 가줘'\" --once"
        )

    # ──────────────────────────────────────────────────────────────────────
    # 모델 warmup
    # ──────────────────────────────────────────────────────────────────────
    def _warmup(self):
        dummy = PILImage.fromarray(np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8))
        inputs = self._build_vlm_inputs(dummy, "/no_think Output: {\"steering\":0,\"speed\":0,\"goal_reached\":false}")
        with torch.inference_mode():
            self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                do_sample=False, use_cache=True)
        self.get_logger().info("Warmup done.")

    # ──────────────────────────────────────────────────────────────────────
    # 콜백
    # ──────────────────────────────────────────────────────────────────────
    def _goal_cmd_cb(self, msg: String):
        parsed = parse_goal_cmd(msg.data)
        action = parsed["action"]

        if action == "stop":
            self.is_stopped = True
            self.steering = 0; self.speed = 0
            self.get_logger().info("⛔ 정지 명령 수신")
            return

        if action == "resume":
            self.is_stopped = False
            self.get_logger().info("▶ 재개 명령 수신")
            return

        # "go"
        self.goal = {"target": parsed["target"], "mode": parsed["mode"]}
        self.goal_reached = False
        self.is_stopped   = False
        wp = WAYPOINTS.get(self.goal["target"], (0, 0))
        self.get_logger().info(
            f"🎯 새 목표: {self.goal['target']} ({wp[0]:.1f}, {wp[1]:.1f}) "
            f"| 모드: {self.goal['mode']}"
        )

    def _image_cb(self, msg: Image):
        self.latest_image = msg
        if (not self.is_inferring and self.goal is not None
                and not self.is_stopped and not self.goal_reached):
            threading.Thread(target=self._run_inference, daemon=True).start()

    def _lidar_cb(self, msg: LaserScan):
        """전방 ±30° 최소 거리 계산 (차체 제외: 0.8m 이상만 고려)."""
        ranges = np.array(msg.ranges, dtype=float)
        n = len(ranges)
        if n == 0:
            return
        # 전방 ±30° 인덱스
        step = max(1, int(round(np.radians(30) / msg.angle_increment))) if msg.angle_increment > 0 else 30
        front_idx = list(range(0, step + 1)) + list(range(n - step, n))
        front_ranges = ranges[front_idx]
        # 차체 자기 감지 제거: 유효 범위 0.8~12m
        # 1.0m 미만은 차체 자신으로 제외 (차체 최대 ~0.86m)
        valid = front_ranges[(np.isfinite(front_ranges)) & (front_ranges >= 1.0) & (front_ranges <= 12.0)]
        self.obstacle_min_dist = float(np.min(valid)) if len(valid) > 0 else 999.0

    def _odom_cb(self, msg: Odometry):
        """/odom 에서 자차 위치·방향 취득."""
        q = msg.pose.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        with self.pose_lock:
            self.car_x = msg.pose.pose.position.x
            self.car_y = msg.pose.pose.position.y
            self.car_yaw = yaw

    # ──────────────────────────────────────────────────────────────────────
    # 추론 (카메라 콜백에서 스레드로 실행)
    # ──────────────────────────────────────────────────────────────────────
    def _run_inference(self):
        with self.infer_lock:
            if self.is_inferring:
                return
            self.is_inferring = True
        try:
            if self.latest_image is None or self.goal is None:
                return

            from cv_bridge import CvBridge
            bridge = CvBridge()
            cv_img = bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')
            cv_img = cv2.resize(cv_img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
            pil_img = PILImage.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

            target = self.goal["target"]
            mode   = self.goal["mode"]
            wp     = WAYPOINTS.get(target, (0, 0))

            # ── direct 모드: 기하학적 조향 + VLM 보조 ──
            with self.pose_lock:
                car_x_snap  = self.car_x
                car_y_snap  = self.car_y
                car_yaw_snap = self.car_yaw
            if mode == "direct" and car_x_snap is not None:
                dx = wp[0] - car_x_snap
                dy = wp[1] - car_y_snap
                dist = math.sqrt(dx*dx + dy*dy)

                # 목표 도달 판정
                if dist < GOAL_REACHED_DIST:
                    self.goal_reached = True
                    self.steering = 0; self.speed = 0
                    self.get_logger().info(f"✅ 목표 도달: {target} (dist={dist:.2f}m)")
                    self._publish_status(f"REACHED:{target}")
                    return

                # 기하학적 heading 계산
                # ※ 이 차량 모델의 forward 방향 = yaw - π/2
                #   (prius_hybrid: body +X가 world에서 yaw=0 방향)
                car_forward     = car_yaw_snap - math.pi / 2
                target_heading  = math.atan2(dy, dx)
                heading_error   = normalize_angle(target_heading - car_forward)
                heading_err_deg = math.degrees(heading_error)

                abs_err_deg = abs(heading_err_deg)

                # 비례 조향 (-7 ~ +7)
                # heading_error > 0 = 목표가 LEFT → 음수 조향(좌회전)
                geo_steering = int(np.clip(-heading_error * HEADING_GAIN, -7, 7))

                # ── 헤딩 기반 속도 제한 ────────────────────────────────────────
                # Ackermann 조향: 전진 속도 없이는 회전 불가 → 최소 속도 유지
                # 큰 heading error 시 저속 최대 조향으로 빠르게 정렬
                if abs_err_deg > 60.0:
                    effective_speed = 30   # 저속 아커만 선회 (최대 조향)
                elif abs_err_deg > 30.0:
                    effective_speed = 40   # 중속 + 회전
                else:
                    effective_speed = 55   # 정상 주행 (lidar 감쇄 별도)

                # 방향이 어느 정도 맞을 때만 VLM 실행 (FPS 절약 + 장애물 인식)
                vlm_reached = False
                if abs_err_deg < 45.0:
                    prompt = make_direct_prompt(target, heading_err_deg, dist)
                    _, vlm_sp_aux, vlm_reached = self._vlm_infer(pil_img, prompt)
                    # VLM이 stop을 권고하면 (신호등·장애물) 속도 제한
                    if vlm_sp_aux < effective_speed:
                        effective_speed = vlm_sp_aux

                if vlm_reached and dist < 4.0:
                    self.goal_reached = True
                    geo_steering = 0; effective_speed = 0
                    self.get_logger().info(f"✅ VLM 도달 감지: {target}")

                self._update_action(geo_steering, effective_speed, dist)
                self.get_logger().info(
                    f"[direct] → {target}  dist={dist:.1f}m  "
                    f"heading_err={heading_err_deg:+.1f}°  "
                    f"geo_st={geo_steering}  sp={effective_speed}"
                    f"  ({'TURN' if abs_err_deg > 60 else 'CREEP' if abs_err_deg > 30 else 'CRUISE'})"
                )

            # ── track 모드: VLM 순수 차선 추종 ──
            else:
                goal_descs = {
                    "traffic_light": "Go to the traffic light following the road/track",
                    "start":         "Return to starting position following the road",
                    "obstacle_1":    "Navigate to obstacle zone 1 following the road",
                    "obstacle_2":    "Navigate to obstacle zone 2 following the road",
                    "mid":           "Navigate to the midpoint of the track",
                }
                desc   = goal_descs.get(target, f"Navigate to {target} following the track")
                prompt = make_track_prompt(target, desc)
                vlm_st, vlm_sp, vlm_reached = self._vlm_infer(pil_img, prompt)

                if vlm_reached:
                    self.goal_reached = True
                    vlm_st = 0; vlm_sp = 0
                    self.get_logger().info(f"✅ VLM 도달 감지: {target}")

                self._update_action(vlm_st, vlm_sp, None)
                self.get_logger().info(
                    f"[track] → {target}  st={vlm_st}  sp={vlm_sp}  "
                    f"reached={vlm_reached}"
                )

        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
        finally:
            self.is_inferring = False

    # ──────────────────────────────────────────────────────────────────────
    # VLM 추론 헬퍼
    # ──────────────────────────────────────────────────────────────────────
    def _build_vlm_inputs(self, pil_img, prompt: str) -> dict:
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": pil_img},
            {"type": "text",  "text":  prompt},
        ]}]
        text = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        return self.processor(
            text=[text], images=[pil_img], padding=True, return_tensors="pt"
        ).to("cuda:0")

    def _vlm_infer(self, pil_img, prompt: str) -> tuple[int, int, bool]:
        """(steering, speed, goal_reached) 반환."""
        inputs = self._build_vlm_inputs(pil_img, prompt)
        t0 = time.monotonic()
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, use_cache=True,
            )
        elapsed = time.monotonic() - t0
        self._infer_times.append(elapsed)
        if len(self._infer_times) > 10:
            self._infer_times.pop(0)

        new_tok  = out[0][inputs.input_ids.shape[1]:]
        response = self.processor.decode(new_tok, skip_special_tokens=True).strip()

        m = re.search(r'\{[^}]+\}', response)
        if not m:
            return 0, 50, False
        try:
            d = json.loads(m.group())
            st      = max(-7, min(7,   int(d.get("steering",    0))))
            sp      = max(0,  min(100, int(d.get("speed",       50))))
            reached = bool(d.get("goal_reached", False))
            return st, sp, reached
        except Exception:
            return 0, 50, False

    # ──────────────────────────────────────────────────────────────────────
    # 장애물 속도 보정 적용
    # ──────────────────────────────────────────────────────────────────────
    def _update_action(self, steering: int, speed: int, dist_to_goal):
        self.steering = int(np.clip(steering, -7, 7))

        # 장애물 속도 감쇄 적용
        factor = obstacle_speed_factor(self.obstacle_min_dist)
        safe_speed = int(round(speed * factor))

        if factor < 1.0:
            self.get_logger().warn(
                f"⚠️ 장애물 {self.obstacle_min_dist:.1f}m → "
                f"속도 {speed} × {factor:.1f} = {safe_speed}"
            )
        self.speed = safe_speed

    # ──────────────────────────────────────────────────────────────────────
    # 퍼블리시 (10 Hz 타이머)
    # ──────────────────────────────────────────────────────────────────────
    def _publish_cb(self):
        if self.is_stopped or self.goal_reached:
            self.steering = 0; self.speed = 0

        msg = MotionCommand()
        msg.steering    = self.steering
        msg.left_speed  = self.speed
        msg.right_speed = self.speed
        self.pub.publish(msg)

    def _publish_status(self, status: str):
        self.status_pub.publish(String(data=status))


def main(args=None):
    rclpy.init(args=args)
    node = VLABrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

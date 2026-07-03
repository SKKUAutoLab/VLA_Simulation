#!/usr/bin/env python3
"""
VLA Agent Node — MCP 스타일 순수 VLA 직접 제어
==============================================
하이브리드(기하 제어) 없이 VLM이 차량을 직접 제어한다.

MCP 매핑:
  Resources(컨텍스트) : track_features.json (주차칸/IN·OUT/횡단보도/장애물/
                        신호등/출발점 좌표) + 현재 차량 상태
  Tools               : drive(steering, speed) · stop() · arrived(note)
  Client/Agent        : Qwen3-VL — 이미지+컨텍스트를 보고 매 사이클 tool 호출

조향·속도의 모든 값은 VLM이 결정한다. 본 노드는 tool 실행기 + 10Hz ZOH
퍼블리셔 + 최소 LiDAR 안전정지(주행 판단 아님, 안전 인터록)만 담당한다.

명령:
    ros2 topic pub /vla/goal_cmd std_msgs/msg/String "data: '신호등까지 가줘'" --once
    ros2 topic pub /vla/goal_cmd std_msgs/msg/String "data: 'P2에 주차해'" --once
    "멈춰"/"정지" → 정지,  "출발"/"계속" → 재개

한계(정직):
  - Qwen3-VL-2B 추론 지연(수백 ms)으로 제어 루프가 느림 → 저속 강제.
  - 제로샷 2B 조향은 흔들릴 수 있음. 주차는 신뢰도 낮음.
  - 정밀도 부족 시: track_features.json + 기하 expert로 데모 생성 → LoRA 파인튜닝.
"""

import os
import re
import json
import math
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy,
                       QoSReliabilityPolicy)
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from interfaces_pkg.msg import MotionCommand

import cv2
import numpy as np
from PIL import Image as PILImage
import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

# ─── 설정 ──────────────────────────────────────────────────────────────────
MODEL_NAME      = "Qwen/Qwen3-VL-2B-Instruct"
CAMERA_TOPIC    = "camera/image_raw"
CONTROL_TOPIC   = "topic_control_signal"
GOAL_CMD_TOPIC  = "vla/goal_cmd"
LIDAR_TOPIC     = "lidar_processed"
ODOM_TOPIC      = "/odom"
FEATURES_PATH   = os.path.expanduser("~/track_features.json")

PUBLISH_HZ       = 10.0
# 해상도↓ + 출력토큰↓ 로 ≥10FPS 확보 (5090 실측: 320x240+terse ≈ 11.4FPS).
# 정밀도 필요 시 640x480까지 올릴 수 있으나 FPS 하락(≈8.9FPS).
INPUT_W, INPUT_H = 320, 240
PROCESSOR_PIXELS = INPUT_W * INPUT_H
MAX_NEW_TOKENS   = 10          # terse 액션 출력용 (디코드 토큰이 지연의 주범)

MAX_SPEED        = 55          # 지연 보정용 속도 상한
ESTOP_DIST       = 1.0         # LiDAR 안전정지 거리 [m] (안전 인터록)
GOAL_REACHED_DIST = 3.0

# Prius yaw offset (검증됨): body +X = (yaw - π/2) 방향
YAW_OFFSET = math.pi / 2

# 상태 컨텍스트(차량 pose→목표 bearing/거리)를 VLM에 제공할지.
# False면 순수 비전(이미지만)으로 운전 — 2B 제로샷엔 훨씬 어려움.
USE_STATE_CONTEXT = True

# 기본 웨이포인트 (features에 없는 보조 지점)
BASE_WAYPOINTS = {
    "start": (-2.55, -22.71),
    "mid":   (-2.55,  -2.00),
}


# ─── 유틸 ─────────────────────────────────────────────────────────────────
def quat_to_yaw(qx, qy, qz, qw) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def normalize_angle(a: float) -> float:
    while a > math.pi:  a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


def load_targets(path: str):
    """track_features.json → {name: {"xy":(x,y), "yaw":옵션, "kind":...}}."""
    targets = {k: {"xy": v, "kind": "waypoint"} for k, v in BASE_WAYPOINTS.items()}
    if not os.path.exists(path):
        return targets, False
    try:
        data = json.load(open(path))
    except Exception:
        return targets, False
    feats = data.get("features", {})

    def _xy(d):
        if "center_world" in d: return tuple(d["center_world"])
        if "point_world"  in d: return tuple(d["point_world"])
        return None

    for kind in ("vertical_parking", "parallel_parking"):
        for s in feats.get(kind, []):
            xy = _xy(s)
            if xy:
                tp = s.get("target_poses", {})
                targets[s["id"].lower()] = {
                    "xy": xy, "kind": kind,
                    "front_yaw": tp.get("front", {}).get("base_pose", {}).get("yaw"),
                    "rear_yaw":  tp.get("rear",  {}).get("base_pose", {}).get("yaw"),
                }
    for kind in ("in_line", "out_line", "crosswalk"):
        items = feats.get(kind, [])
        items = items if isinstance(items, list) else [items]
        for i, s in enumerate(items):
            xy = _xy(s)
            if xy:
                targets[kind if i == 0 else f"{kind}{i}"] = {"xy": xy, "kind": kind}
    for kind in ("obstacle", "traffic_light", "start_point"):
        for s in feats.get(kind, []):
            xy = _xy(s)
            if xy:
                targets[s["id"].lower()] = {"xy": xy, "kind": kind}
    # 별칭
    if "tl1" in targets:
        targets["traffic_light"] = targets["tl1"]
    return targets, True


def resolve_target(text: str, targets: dict):
    """자연어 → 목표 키. 못 찾으면 None."""
    t = text.strip().lower()
    alias = {
        "신호등": "traffic_light", "traffic": "traffic_light",
        "출발점": "start", "처음": "start", "원점": "start", "스폰": "start",
        "중간": "mid",
        "횡단보도": "crosswalk", "crosswalk": "crosswalk",
    }
    for k, v in alias.items():
        if k in t and v in targets:
            return v
    # 주차칸/장애물 id 직접 매칭 (p1..v4, ob_*, tl1, in1..out4)
    for key in targets:
        if key in t:
            return key
    return None


# ─── Tool 스키마 (terse — 입력/출력 토큰 최소화로 ≥10FPS, 5090 실측 11.0FPS) ──
# MCP tool 개념 유지(drive/stop/arrived)하되 와이어 포맷만 짧게.
TOOL_SPEC = (
    "Drive a car using the front camera. Reply ONE line only:\n"
    "D <st> <sp>  st=-7..7 (NEGATIVE=left, POSITIVE=right), sp=0..100; "
    "S=stop; A=arrived. Ex: D -3 40"
)


# ─── 메인 노드 ────────────────────────────────────────────────────────────
class VLAAgentNode(Node):
    def __init__(self):
        super().__init__("vla_agent_node")

        # 목표 / 상태
        self.targets, ok = load_targets(FEATURES_PATH)
        self.goal_name = None          # 목표 키
        self.goal_mode = "track"       # "track"=도로 따라가기 | "direct"=직선 무시
        self.is_stopped = False
        self.goal_reached = False

        self.car_x = self.car_y = self.car_yaw = None
        self.pose_lock = threading.Lock()
        self.obstacle_min_dist = 999.0

        self.latest_image = None
        self.is_inferring = False
        self.infer_lock = threading.Lock()
        self._bridge = None

        # 제어 출력 (ZOH 유지)
        self.steering = 0
        self.speed = 0
        self._infer_times = []

        qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         durability=QoSDurabilityPolicy.VOLATILE, depth=1)

        # 모델
        self.get_logger().info(f"Loading {MODEL_NAME} ...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16,
            device_map="cuda:0", attn_implementation="sdpa")
        self.model.eval()
        self.processor = Qwen3VLProcessor.from_pretrained(
            MODEL_NAME, min_pixels=PROCESSOR_PIXELS, max_pixels=PROCESSOR_PIXELS)
        self._warmup()
        self.get_logger().info(
            f"VLA Agent ready. targets={len(self.targets)} "
            f"(features={'loaded' if ok else 'MISSING'})  "
            f"VRAM={torch.cuda.memory_allocated()//1024**2}MB")

        # 구독/발행
        self.create_subscription(Image, CAMERA_TOPIC, self._image_cb, qos)
        self.create_subscription(String, GOAL_CMD_TOPIC, self._goal_cmd_cb, qos)
        self.create_subscription(LaserScan, LIDAR_TOPIC, self._lidar_cb, qos)
        self.create_subscription(
            Odometry, ODOM_TOPIC, self._odom_cb,
            QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                       history=QoSHistoryPolicy.KEEP_LAST,
                       durability=QoSDurabilityPolicy.VOLATILE))
        self.pub = self.create_publisher(MotionCommand, CONTROL_TOPIC, qos)
        self.status_pub = self.create_publisher(String, "vla/status", qos)
        self.timer = self.create_timer(1.0 / PUBLISH_HZ, self._publish_cb)
        self.get_logger().info("Waiting for goal command...")

    # ── warmup ──
    def _warmup(self):
        dummy = PILImage.fromarray(np.zeros((INPUT_H, INPUT_W, 3), np.uint8))
        inputs = self._build_inputs(dummy, 'Output: {"tool":"stop","args":{}}')
        with torch.inference_mode():
            self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                do_sample=False, use_cache=True)
        self.get_logger().info("Warmup done.")

    # ── 콜백 ──
    def _goal_cmd_cb(self, msg: String):
        t = msg.text if hasattr(msg, "text") else msg.data
        low = t.strip().lower()
        if any(k in low for k in ("멈춰", "정지", "stop", "멈춤")):
            self.is_stopped = True
            self.steering = 0; self.speed = 0
            self.get_logger().info("⏸️  정지")
            return
        if any(k in low for k in ("출발", "계속", "resume", "go", "시작")) \
                and resolve_target(low, self.targets) is None:
            self.is_stopped = False
            self.get_logger().info("▶️  재개")
            return
        tgt = resolve_target(low, self.targets)
        if tgt is None:
            self.get_logger().warn(f"목표를 못 찾음: '{t}' (기본 traffic_light)")
            tgt = "traffic_light" if "traffic_light" in self.targets else None
        # 주행 모드: '직접/바로/무시/direct/straight' → 직선 모드, 아니면 트랙 모드
        mode = ("direct" if any(k in low for k in
                ("직접", "바로", "무시", "direct", "straight")) else "track")
        self.goal_name = tgt
        self.goal_mode = mode
        self.is_stopped = False
        self.goal_reached = False
        xy = self.targets.get(tgt, {}).get("xy")
        self.get_logger().info(f"🎯 목표: {tgt} {xy}  모드: {mode}")

    def _image_cb(self, msg: Image):
        self.latest_image = msg
        if (not self.is_inferring and self.goal_name is not None
                and not self.is_stopped and not self.goal_reached):
            threading.Thread(target=self._run_inference, daemon=True).start()

    def _lidar_cb(self, msg: LaserScan):
        """전방 ±30° 최소 거리. 차체 자기감지 제외(1.0m 미만 무시)."""
        ranges = np.array(msg.ranges, dtype=float)
        n = len(ranges)
        if n == 0:
            return
        step = (max(1, int(round(math.radians(30) / msg.angle_increment)))
                if msg.angle_increment > 0 else 30)
        front_idx = list(range(0, step + 1)) + list(range(max(0, n - step), n))
        front = ranges[front_idx]
        valid = front[np.isfinite(front) & (front >= 1.0) & (front <= 12.0)]
        self.obstacle_min_dist = float(np.min(valid)) if len(valid) else 999.0

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        with self.pose_lock:
            self.car_x, self.car_y = p.x, p.y
            self.car_yaw = quat_to_yaw(o.x, o.y, o.z, o.w)

    # ── 컨텍스트 빌드 (MCP resources) ── 모드별 주행 지시 ──────────────────────
    def _build_context(self):
        """현재 목표+차량 상태+모드별 지시를 간결한 텍스트로 (FPS 위해 짧게)."""
        g = self.targets.get(self.goal_name, {})
        gx, gy = g.get("xy", (None, None))
        seg = f"Goal {self.goal_name}"
        if gx is not None:
            seg += f" at ({gx:.1f},{gy:.1f})"
        seg += "."

        # 목표 방위 (bearing) 텍스트
        bearing_txt = ""
        if USE_STATE_CONTEXT and gx is not None:
            with self.pose_lock:
                cx, cy, cyaw = self.car_x, self.car_y, self.car_yaw
            if cx is not None:
                dx, dy = gx - cx, gy - cy
                dist = math.hypot(dx, dy)
                # heading_error>0 = 목표가 LEFT (검증된 geo=-heading_error 기준)
                bearing = math.degrees(
                    normalize_angle(math.atan2(dy, dx) - (cyaw - YAW_OFFSET)))
                # bearing각 → 추천 조향 크기 (LEFT=음수, RIGHT=양수)
                ab = abs(bearing)
                mag = 0 if ab <= 5 else (3 if ab <= 15 else (5 if ab <= 30 else 7))
                if mag == 0:
                    side, st_sug, sp_sug = "STRAIGHT", 0, 50
                else:
                    side = "LEFT" if bearing > 0 else "RIGHT"
                    st_sug = -mag if bearing > 0 else mag
                    sp_sug = 40
                # 실측(diag_steering): 단순 "steer POSITIVE" 힌트는 2B가 우회전을
                # 거부(추종 0%). 명령형 + 추천 D 액션을 주면 좌·우·직진 100% 추종.
                bearing_txt = (f" Goal {dist:.0f}m, {ab:.0f}deg to {side}. You MUST "
                               f"steer {side}. Recommended action: D {st_sug} {sp_sug}.")

        if self.goal_mode == "direct":
            # 직선 모드: 차선·차 무시하고 목표로 직진
            seg += bearing_txt + " Head directly to the goal; ignore lanes and cars."
        elif g.get("kind") in ("vertical_parking", "parallel_parking"):
            seg += " Park centered in the slot." + bearing_txt
        else:
            # 트랙 모드: 회색 도로/차선을 따라가고 주차장·잔디·차로 들어가지 않음
            seg += (" Follow the gray ROAD toward the goal. Stay on the road, "
                    "keep between lane lines. Do NOT cut across the parking lot, "
                    "grass, or into other cars. Slow/stop (S) if a car blocks "
                    "the road ahead." + bearing_txt)
        return seg

    # ── 추론 (agent step) ──
    def _run_inference(self):
        with self.infer_lock:
            if self.is_inferring:
                return
            self.is_inferring = True
        try:
            if self.latest_image is None or self.goal_name is None:
                return
            # 도달 판정 (상태 컨텍스트 있을 때 보조)
            g = self.targets.get(self.goal_name, {})
            gx, gy = g.get("xy", (None, None))
            with self.pose_lock:
                cx, cy = self.car_x, self.car_y
            if gx is not None and cx is not None:
                if math.hypot(gx - cx, gy - cy) < GOAL_REACHED_DIST:
                    self._arrive("auto: within goal radius")
                    return

            pil = self._msg_to_pil(self.latest_image)
            ctx = self._build_context()
            prompt = TOOL_SPEC + "\n" + ctx + "\nNext action:"
            tool, args, raw = self._infer_tool(pil, prompt)
            self.get_logger().info(f"[ctx] {ctx}\n[vlm] {raw!r}")
            self._dispatch(tool, args)
        finally:
            self.is_inferring = False

    def _dispatch(self, tool: str, args: dict):
        if tool == "stop":
            self.steering = 0; self.speed = 0
        elif tool == "arrived":
            self._arrive(str(args.get("note", "")))
        elif tool == "drive":
            st = int(max(-7, min(7, args.get("steering", 0))))
            sp = int(max(0, min(100, args.get("speed", 0))))
            self.steering = st
            self.speed = min(sp, MAX_SPEED)
        else:
            self.get_logger().warn(f"알 수 없는 tool: {tool}")
        avg = (sum(self._infer_times) / len(self._infer_times)
               if self._infer_times else 0.0)
        self.get_logger().info(
            f"[agent] {tool}{args}  st={self.steering} sp={self.speed} "
            f"({avg*1000:.0f}ms)")

    def _arrive(self, note: str):
        self.goal_reached = True
        self.steering = 0; self.speed = 0
        self.get_logger().info(f"✅ 도달: {self.goal_name} ({note})")
        self._publish_status(f"REACHED:{self.goal_name}")

    # ── VLM I/O ──
    def _build_inputs(self, pil_img, prompt: str) -> dict:
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": pil_img},
            {"type": "text",  "text":  prompt}]}]
        text = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        return self.processor(text=[text], images=[pil_img],
                              padding=True, return_tensors="pt").to("cuda:0")

    def _infer_tool(self, pil_img, prompt: str):
        inputs = self._build_inputs(pil_img, prompt)
        t0 = time.monotonic()
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                      do_sample=False, use_cache=True)
        self._infer_times.append(time.monotonic() - t0)
        if len(self._infer_times) > 10:
            self._infer_times.pop(0)
        new_tok = out[0][inputs.input_ids.shape[1]:]
        resp = self.processor.decode(new_tok, skip_special_tokens=True).strip()
        tool, args = parse_tool_call(resp)
        return tool, args, resp

    # ── 발행 (10Hz ZOH + 안전 인터록) ──
    def _publish_cb(self):
        steering, speed = self.steering, self.speed
        if self.is_stopped or self.goal_reached:
            steering, speed = 0, 0
        # 전방 장애물은 카메라(VLM)가 시각으로 처리. LiDAR는 후방 마운트라
        # 전방 e-stop에 부적합 → 사용 안 함.
        msg = MotionCommand()
        msg.steering = int(steering)
        msg.left_speed = int(speed)
        msg.right_speed = int(speed)
        self.pub.publish(msg)

    def _publish_status(self, status: str):
        self.status_pub.publish(String(data=status))

    def _msg_to_pil(self, msg: Image) -> PILImage.Image:
        if self._bridge is None:
            from cv_bridge import CvBridge
            self._bridge = CvBridge()
        cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv_img = cv2.resize(cv_img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
        return PILImage.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def parse_tool_call(resp: str):
    """모델 출력 → (tool, args). terse(D/S/A) 우선, JSON 폴백. 실패 시 ('stop',{})."""
    s = resp.strip()
    # ── terse 포맷 ────────────────────────────────────────────────────────
    up = s.lstrip("`*: ").upper()
    if up[:1] == "A":
        return "arrived", {"note": "vlm"}
    if up[:1] == "S" and not re.match(r'S?\s*-?\d', up):
        return "stop", {}
    if up[:1] == "D":
        s = s[1:]
    nums = re.findall(r'-?\d+', s)
    if nums:
        st = int(nums[0])
        sp = int(nums[1]) if len(nums) > 1 else 0
        return "drive", {"steering": st, "speed": sp}
    # ── JSON 폴백 (구버전/혼합 출력 호환) ──────────────────────────────────
    m = re.search(r'\{.*\}', resp, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group())
            tool = d.get("tool", "stop")
            args = d.get("args", {}) if isinstance(d.get("args"), dict) else {}
            if tool == "stop" and ("steering" in d or "speed" in d):
                tool, args = "drive", {"steering": d.get("steering", 0),
                                       "speed": d.get("speed", 0)}
            return tool, args
        except Exception:
            pass
    return "stop", {}


def main(args=None):
    rclpy.init(args=args)
    node = VLAAgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

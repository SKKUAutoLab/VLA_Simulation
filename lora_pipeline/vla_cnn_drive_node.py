#!/usr/bin/env python3
"""
VLA 명령층 + CNN 비전주행 통합 노드
===================================
- VLA(Qwen) : 자연어 명령 해석 → 목표/모드 (두뇌)
- CNN       : 카메라 → 조향 (track 모드 저수준 주행)
- 기하       : /odom 기반 직선 조향 (direct 모드)
- 목표 좌표 근접 시 정지.

명령 토픽:
    ros2 topic pub /vla/command std_msgs/msg/String "data: '신호등으로 가줘'" --once   # track
    ros2 topic pub /vla/command std_msgs/msg/String "data: '신호등으로 직진해서 가줘'" --once # direct
    "멈춰"/"출발"

실행:
    ros2 launch simulation_pkg teleop_sim.launch.py
    python3 lora_pipeline/vla_cnn_drive_node.py
"""
import os, re, json, math, threading
import cv2
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy,
                       QoSDurabilityPolicy)
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from interfaces_pkg.msg import MotionCommand
from train_cnn import preprocess
from train_cnn_lane import LaneCNN

HERE = os.path.dirname(__file__)
CNN_MODEL = os.environ.get("CNN_MODEL", os.path.join(HERE, "cnn_lane_model.pt"))
QWEN = "Qwen/Qwen3-VL-2B-Instruct"
CAMERA_TOPIC, CONTROL_TOPIC, ODOM_TOPIC = "camera/image_raw", "topic_control_signal", "/odom"

WAYPOINTS = {"traffic_light": (-5.63, 17.9), "start": (-2.55, -22.71)}
GOAL_REACHED_DIRECT = 3.5  # 직선: 목표점 근접(거리)
TRACK_REACH_IDX = 14       # 차도: 신호등 도로구간(중심선 인덱스) ±N 도달 시(차선 무관)
CENTERLINE = os.path.expanduser("~/track_gt_manual.json")
YAW_OFFSET = math.pi / 2
HEADING_GAIN = 4.5
CRUISE = int(os.environ.get("CRUISE", "120"))       # 0~255 (MAX_SPEED 5m/s → ~2.35m/s)
CRUISE_TURN = int(os.environ.get("CRUISE_TURN", "70"))  # 큰 조향/목표근접 감속
STEER_SCALE = float(os.environ.get("STEER_SCALE", "2.0"))  # 과소조향 보정 배율(실측 2.0서 1차선 이탈 1.2m)


def quat_to_yaw(qx, qy, qz, qw):
    return math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))


def normalize_angle(a):
    while a > math.pi:  a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a


def keyword_parse(text):
    """Qwen 실패 시 폴백."""
    t = text.strip().lower()
    if any(k in t for k in ("멈춰", "정지", "stop", "멈춤")):
        return {"goal": None, "mode": None, "action": "stop"}
    goal = "traffic_light" if any(k in t for k in ("신호등", "traffic")) else (
           "start" if any(k in t for k in ("출발점", "처음", "원점", "스폰")) else None)
    if goal is None and any(k in t for k in ("출발", "계속", "resume", "go")):
        return {"goal": None, "mode": None, "action": "resume"}
    mode = "direct" if any(k in t for k in ("직진", "바로", "무시", "직선", "direct", "straight")) else "track"
    return {"goal": goal or "traffic_light", "mode": mode, "action": "go"}


class VLACNNNode(Node):
    def __init__(self):
        super().__init__("vla_cnn_drive_node")
        # CNN
        ck = torch.load(CNN_MODEL, map_location="cuda:0")
        self.cnn = LaneCNN().to("cuda:0")
        self.cnn(torch.zeros(1, 3, ck["in_h"], ck["in_w"], device="cuda:0"),
                 torch.zeros(1, dtype=torch.long, device="cuda:0"))
        self.cnn.load_state_dict(ck["state_dict"]); self.cnn.eval()
        self.smax = ck["steer_max"]
        self.lane = 0   # 0=1차선(inner), 1=2차선(outer)
        # Qwen (명령해석). NO_QWEN=1 이면 건너뜀(키워드만으로 차선/랩 결정 — eval용)
        self.qwen = self.qproc = None
        if os.environ.get("NO_QWEN") != "1":
            self.get_logger().info("Loading Qwen for command parsing...")
            from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
            self.qproc = Qwen3VLProcessor.from_pretrained(QWEN)
            self.qwen = Qwen3VLForConditionalGeneration.from_pretrained(
                QWEN, torch_dtype=torch.bfloat16, device_map="cuda:0",
                attn_implementation="sdpa").eval()

        # 중심선(인덱스 기반 도달용)
        self.cl = [(float(x), float(y)) for x, y in
                   json.load(open(CENTERLINE))["centerline_world"]]
        self.tl_idx = self._nearest_idx(*WAYPOINTS["traffic_light"])

        self.bridge = None
        self.cnn_steer = 0
        self.x = self.y = self.yaw = None
        self.goal = None; self.mode = "track"; self.active = False; self.reached = False
        self.lap = False   # 한바퀴 무정지 차선유지(평가용)
        self.lock = threading.Lock()

        qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         durability=QoSDurabilityPolicy.VOLATILE, depth=1)
        be = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        history=QoSHistoryPolicy.KEEP_LAST, durability=QoSDurabilityPolicy.VOLATILE)
        self.create_subscription(Image, CAMERA_TOPIC, self._img, be)
        self.create_subscription(Odometry, ODOM_TOPIC, self._odom, be)
        self.create_subscription(String, "vla/command", self._cmd, qos)
        self.pub = self.create_publisher(MotionCommand, CONTROL_TOPIC, qos)
        self.status_pub = self.create_publisher(String, "vla/status", qos)
        self.create_timer(0.05, self._control)
        self.get_logger().info("VLA+CNN ready. 명령 대기: /vla/command")

    # ── VLA 명령 해석 ──
    def _vla_interpret(self, text):
        prompt = (f"Driving command: '{text}'\n"
                  "Reply EXACTLY one line: GOAL=<traffic_light|start|none> "
                  "MODE=<track|direct> ACT=<go|stop|resume>\n"
                  "Rules: '직진/바로/무시/직선/straight/direct'->MODE=direct else track; "
                  "'멈춰/정지/stop'->ACT=stop; '출발/계속'(목표없이)->ACT=resume; "
                  "신호등->GOAL=traffic_light.")
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        t = self.qproc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.qproc(text=[t], return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            out = self.qwen.generate(**inp, max_new_tokens=30, do_sample=False)
        resp = self.qproc.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
        g = re.search(r'GOAL=(\w+)', resp); m = re.search(r'MODE=(\w+)', resp); a = re.search(r'ACT=(\w+)', resp)
        if g and m and a:
            goal = g.group(1) if g.group(1) in WAYPOINTS else None
            return {"goal": goal, "mode": m.group(1), "action": a.group(1)}, resp
        return None, resp

    def _cmd(self, msg):
        text = msg.data
        parsed, raw = (self._vla_interpret(text) if self.qwen is not None else (None, ""))
        if parsed is None:
            parsed = keyword_parse(text)
            self.get_logger().warn(f"[VLA] 파싱불명 → 폴백: {parsed}")
        else:
            self.get_logger().info(f"[VLA] '{text}' → {parsed}  (raw:{raw!r})")
        # mode는 키워드로 결정론적 결정(Qwen이 자주 틀림): 직진류 있을 때만 direct
        tl_ = text.lower()
        direct_kw = ("직진", "바로", "무시", "직선", "direct", "straight")
        parsed["mode"] = "direct" if any(k in tl_ for k in direct_kw) else "track"
        # 차선 지정 (없으면 유지)
        if any(k in tl_ for k in ("2차선", "이차선", "바깥", "외측", "outer", "lane 2", "lane2")):
            self.lane = 1
        elif any(k in tl_ for k in ("1차선", "일차선", "안쪽", "내측", "inner", "lane 1", "lane1")):
            self.lane = 0
        self.get_logger().info(f"[lane] {'2차선(outer)' if self.lane==1 else '1차선(inner)'}")
        is_lap = any(k in tl_ for k in ("한바퀴", "한 바퀴", "랩", "lap", "계속 돌"))
        with self.lock:
            act = parsed["action"]
            if act == "stop":
                self.active = False; self.lap = False
            elif act == "resume":
                self.active = True; self.reached = False
            elif is_lap:  # 무정지 한바퀴 차선유지
                self.lap = True; self.active = True; self.reached = False; self.mode = "track"
                self.get_logger().info(f"🔁 LAP 모드 (lane {self.lane})")
            else:  # go
                self.goal = parsed["goal"] or "traffic_light"
                self.mode = parsed["mode"] or "track"
                self.active = True; self.reached = False
                self.get_logger().info(f"🎯 목표={self.goal} {WAYPOINTS[self.goal]} 모드={self.mode}")

    def _odom(self, msg):
        p = msg.pose.pose.position; o = msg.pose.pose.orientation
        with self.lock:
            self.x, self.y = p.x, p.y
            self.yaw = quat_to_yaw(o.x, o.y, o.z, o.w)

    def _img(self, msg):
        if self.bridge is None:
            from cv_bridge import CvBridge; self.bridge = CvBridge()
        bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        x = preprocess(bgr).transpose(2, 0, 1)[None]
        lane_t = torch.tensor([self.lane], dtype=torch.long, device="cuda:0")
        with torch.inference_mode():
            out = self.cnn(torch.from_numpy(x).to("cuda:0"), lane_t).item()
        self.cnn_steer = int(max(-7, min(7, round(out * self.smax * STEER_SCALE))))

    def _nearest_idx(self, x, y):
        return min(range(len(self.cl)),
                   key=lambda k: (self.cl[k][0]-x)**2 + (self.cl[k][1]-y)**2)

    # ── 제어 루프 ──
    def _control(self):
        with self.lock:
            active, reached, goal, mode = self.active, self.reached, self.goal, self.mode
            lap = self.lap
            cx, cy, cyaw = self.x, self.y, self.yaw
        st, sp = 0, 0
        if lap and active:
            # 무정지 한바퀴: CNN 비전 조향, 정지 판정 없음
            st = self.cnn_steer
            sp = CRUISE if abs(st) < 3 else CRUISE_TURN
            m = MotionCommand(); m.steering = int(st); m.left_speed = int(sp); m.right_speed = int(sp)
            self.pub.publish(m); return
        if active and goal is not None and not reached:
            gx, gy = WAYPOINTS[goal]
            dist = math.hypot(gx-cx, gy-cy) if cx is not None else 999
            # 도달 판정: direct=목표점 거리, track=신호등 도로구간 인덱스 도달(차선 무관)
            if mode == "direct":
                reached_now = cx is not None and dist < GOAL_REACHED_DIRECT
            else:
                ci = self._nearest_idx(cx, cy) if cx is not None else -1
                d_idx = (min(abs(ci-self.tl_idx), len(self.cl)-abs(ci-self.tl_idx))
                         if ci >= 0 else 999)
                reached_now = ci >= 0 and d_idx <= TRACK_REACH_IDX
            if reached_now:
                with self.lock: self.reached = True; self.active = False
                self._status(f"REACHED:{goal}")
                self.get_logger().info(f"✅ 도달: {goal} (dist={dist:.2f}m)")
            else:
                if mode == "direct" and cx is not None:
                    car_forward = cyaw - YAW_OFFSET
                    he = normalize_angle(math.atan2(gy-cy, gx-cx) - car_forward)
                    st = int(max(-7, min(7, round(-he*HEADING_GAIN))))
                    sp = CRUISE if abs(math.degrees(he)) < 25 else CRUISE_TURN
                else:  # track = CNN 비전
                    st = self.cnn_steer
                    sp = CRUISE if abs(st) < 3 else CRUISE_TURN
                if dist < 8:
                    sp = CRUISE_TURN
        m = MotionCommand(); m.steering = int(st); m.left_speed = int(sp); m.right_speed = int(sp)
        self.pub.publish(m)

    def _status(self, s):
        self.status_pub.publish(String(data=s))


def main():
    rclpy.init()
    node = VLACNNNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

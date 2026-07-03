#!/usr/bin/env python3
"""
웨이포인트 + pure-pursuit 주행 노드.
이미지+lane_id → WPCNN → 미래 ego 웨이포인트 → pure-pursuit로 조향 계산.
명령: "1차선/2차선"(차선), 그 외는 주행 시작(lane-keep). topic_control_signal 발행.
"""
import os, math, json, re, threading
import cv2
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy)
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from interfaces_pkg.msg import MotionCommand
from train_cnn import preprocess
from train_wp import WPCNN

HERE = os.path.dirname(__file__)
MODEL = os.environ.get("WP_MODEL", os.path.join(HERE, "cnn_wp_model.pt"))
CAMERA_TOPIC, CONTROL_TOPIC, ODOM_TOPIC = "camera/image_raw", "topic_control_signal", "/odom"
# 차선 중심선(바퀴 카운트용): lane0=1차선(중심선), lane1=2차선(바깥)
LANE_FILE = {0: os.path.expanduser("~/track_gt_lane1_demo.json"),
             1: os.path.expanduser("~/track_gt_lane0_demo.json")}
KOR_NUM = {"한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5}
# 실측 튜닝(수동시연 모델, demo-fitted 라인): Ld↓·GAIN↑가 정상상태 오프셋 축소
# 평균이탈 1.0→0.6m. 속도↑ 원하면 CRUISE 올리되 평균 다시 증가 트레이드오프.
# 게인↑/전방주시↓는 평균이탈 줄지만 조향 진동(비틀비틀) 유발 → 완화 + EMA 평활
GAIN = float(os.environ.get("PP_GAIN", "13.0"))    # pure-pursuit 게인(코너 추종 위해 유지)
LOOKAHEAD = float(os.environ.get("PP_LD", "0.55")) # 전방주시[m]
STEER_EMA = float(os.environ.get("STEER_EMA", "0.5"))  # 조향 저역통과(프레임간 점프=비틀비틀 억제)
CRUISE = int(os.environ.get("CRUISE", "70"))
CRUISE_TURN = int(os.environ.get("CRUISE_TURN", "50"))


class WPDriveNode(Node):
    def __init__(self):
        super().__init__("wp_drive_node")
        ck = torch.load(MODEL, map_location="cuda:0")
        self.net = WPCNN(ck["nout"]).to("cuda:0")
        self.net(torch.zeros(1, 3, ck["in_h"], ck["in_w"], device="cuda:0"),
                 torch.zeros(1, dtype=torch.long, device="cuda:0"))
        self.net.load_state_dict(ck["state_dict"]); self.net.eval()
        self.wp_n = ck["wp_n"]; self.wp_scale = ck["wp_scale"]
        self.bridge = None; self.lane = 0; self.active = False
        self.steering = 0; self.speed = 0; self.steer_f = 0.0
        self.lock = threading.Lock()
        # 바퀴 카운트 상태
        self.cl = {k: [(float(a), float(b)) for a, b in json.load(open(p))["centerline_world"]]
                   for k, p in LANE_FILE.items()}
        self.target_laps = 0      # 0=무한(멈출 때까지), N=N바퀴 후 정지
        self.laps_done = 0; self.start_idx = None; self.visited = set(); self.x = self.y = None; self._last = None
        qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST,
                         durability=QoSDurabilityPolicy.VOLATILE, depth=1)
        be = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        history=QoSHistoryPolicy.KEEP_LAST, durability=QoSDurabilityPolicy.VOLATILE)
        self.create_subscription(Image, CAMERA_TOPIC, self._img, be)
        self.create_subscription(Odometry, ODOM_TOPIC, self._odom, be)
        self.create_subscription(String, "vla/command", self._cmd, qos)
        self.pub = self.create_publisher(MotionCommand, CONTROL_TOPIC, qos)
        self.create_timer(0.05, self._pub)
        self.get_logger().info(f"WP drive ready. GAIN={GAIN} Ld={LOOKAHEAD} CRUISE={CRUISE}")

    def _parse_laps(self, t):
        """명령에서 바퀴 수 파싱. 숫자/한·두·세… '바퀴|lap' 있으면 그 수, 없으면 1바퀴 기본.
           '계속/무한/loop forever' → 0(무한)."""
        if any(k in t for k in ("계속", "무한", "forever", "endless")):
            return 0
        m = re.search(r"(\d+)\s*바퀴", t) or re.search(r"(\d+)\s*lap", t)
        if m:
            return int(m.group(1))
        for k, v in KOR_NUM.items():
            if k + "바퀴" in t or k + " 바퀴" in t:
                return v
        if "바퀴" in t or "lap" in t:
            return 1
        return 1   # 주행 명령 기본 = 1바퀴

    def _cmd(self, msg):
        t = msg.data.lower()
        if any(k in t for k in ("2차선", "이차선", "outer", "lane2", "lane 2", "lane two", "second lane")):
            self.lane = 1
        elif any(k in t for k in ("1차선", "일차선", "inner", "lane1", "lane 1", "lane one", "first lane")):
            self.lane = 0
        if any(k in t for k in ("멈춰", "정지", "stop")):
            with self.lock:
                self.active = False; self.target_laps = 0; self._last = None
            self.get_logger().info("[wp] 정지")
            return
        laps = self._parse_laps(t)
        key = (self.lane, laps)
        if self.active and getattr(self, "_last", None) == key:   # 동일 명령 연속재전송 무시(카운트 리셋 방지)
            return
        with self.lock:
            self._last = key
            self.active = True; self.target_laps = laps
            self.laps_done = 0; self.start_idx = None; self.visited = set()
        self.get_logger().info(f"[wp] lane={self.lane} 시작 ({'무한' if laps==0 else str(laps)+'바퀴'})")

    def _odom(self, msg):
        p = msg.pose.pose.position
        with self.lock:
            self.x, self.y = p.x, p.y
            if not self.active or self.target_laps == 0:
                return
            cl = self.cl[self.lane]; N = len(cl)
            i = min(range(N), key=lambda k: (cl[k][0]-p.x)**2 + (cl[k][1]-p.y)**2)
            if self.start_idx is None:
                self.start_idx = i
            self.visited.add(i // 20)
            # 한 바퀴 = 거의 전 구간 방문 + 출발 인덱스 부근 복귀(순환거리)
            d = abs(i - self.start_idx); d = min(d, N - d)
            if len(self.visited) >= 34 and d < 12:
                self.laps_done += 1
                self.get_logger().info(f"🔁 {self.laps_done}/{self.target_laps}바퀴 완료")
                self.visited = set(); self.start_idx = i
                if self.laps_done >= self.target_laps:
                    self.active = False
                    self.get_logger().info(f"✅ {self.target_laps}바퀴 완료 — 정지")

    def _img(self, msg):
        if self.bridge is None:
            from cv_bridge import CvBridge; self.bridge = CvBridge()
        bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        x = preprocess(bgr).transpose(2, 0, 1)[None]
        lane_t = torch.tensor([self.lane], dtype=torch.long, device="cuda:0")
        with torch.inference_mode():
            out = self.net(torch.from_numpy(x).to("cuda:0"), lane_t)[0].cpu().numpy() * self.wp_scale
        # ego 웨이포인트 (ex,ey) 쌍
        pts = [(out[2*k], out[2*k+1]) for k in range(self.wp_n)]
        # pure-pursuit: 전방거리 LOOKAHEAD 근처 점 선택
        target = None
        for ex, ey in pts:
            if ex >= LOOKAHEAD:
                target = (ex, ey); break
        if target is None:
            target = pts[-1]
        ex, ey = target
        he = math.atan2(ey, max(0.3, ex))   # 좌(+)면 좌회전 필요
        raw = -he * GAIN                     # 음수=좌 규약
        self.steer_f = STEER_EMA * raw + (1 - STEER_EMA) * self.steer_f   # 시간평활(비틀비틀 억제)
        st = int(max(-7, min(7, round(self.steer_f))))
        with self.lock:
            self.steering = st
            self.speed = CRUISE if abs(st) < 3 else CRUISE_TURN

    def _pub(self):
        with self.lock:
            st, sp = (self.steering, self.speed) if self.active else (0, 0)
        m = MotionCommand(); m.steering = int(st); m.left_speed = int(sp); m.right_speed = int(sp)
        self.pub.publish(m)


def main():
    rclpy.init(); node = WPDriveNode()
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

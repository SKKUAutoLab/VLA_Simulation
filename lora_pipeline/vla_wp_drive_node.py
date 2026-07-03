#!/usr/bin/env python3
"""
순수 VLA 주행 노드 (B) — Qwen3-VL 비전 인코더(1-pass) → WP 회귀 헤드 → pure-pursuit.
토큰생성 없음(빠름). 명령: "1차선/2차선/lane one|two", "N바퀴/계속", "멈춰". 바퀴 자동정지.
"""
import os, math, json, re, threading, time
import cv2, numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy)
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from interfaces_pkg.msg import MotionCommand
from vla_vision import load_vision, extract_feature
from train_vla_wp import Head

HERE = os.path.dirname(__file__)
HEAD_PT = os.environ.get("VLA_HEAD", os.path.join(HERE, "vla_wp_head.pt"))
CAMERA_TOPIC, CONTROL_TOPIC, ODOM_TOPIC = "camera/image_raw", "topic_control_signal", "/odom"
LANE_FILE = {0: os.path.expanduser("~/track_gt_lane1_demo.json"),
             1: os.path.expanduser("~/track_gt_lane0_demo.json")}
KOR_NUM = {"한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5}
GAIN = float(os.environ.get("PP_GAIN", "13.0"))
LOOKAHEAD = float(os.environ.get("PP_LD", "0.55"))
STEER_EMA = float(os.environ.get("STEER_EMA", "0.5"))
CRUISE = int(os.environ.get("CRUISE", "70"))
CRUISE_TURN = int(os.environ.get("CRUISE_TURN", "50"))


class VLADriveNode(Node):
    def __init__(self):
        super().__init__("vla_wp_drive_node")
        self.get_logger().info("Qwen3-VL 비전 인코더 로딩...")
        self.vis, self.proc = load_vision()
        ck = torch.load(HEAD_PT, map_location="cuda:0")
        self.head = Head(ck["nout"]).to("cuda:0"); self.head.load_state_dict(ck["state_dict"]); self.head.eval()
        self.wp_n = ck["wp_n"]; self.wp_scale = ck["wp_scale"]
        self.bridge = None; self.lane = 0; self.active = False
        self.steering = 0; self.speed = 0; self.steer_f = 0.0
        self.inferring = False; self.latest = None
        self.lock = threading.Lock()
        self.cl = {k: [(float(a), float(b)) for a, b in json.load(open(p))["centerline_world"]]
                   for k, p in LANE_FILE.items()}
        self.target_laps = 0; self.laps_done = 0; self.start_idx = None; self.visited = set()
        self.fps = 0.0; self._last = None
        qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST,
                         durability=QoSDurabilityPolicy.VOLATILE, depth=1)
        be = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        history=QoSHistoryPolicy.KEEP_LAST, durability=QoSDurabilityPolicy.VOLATILE)
        self.create_subscription(Image, CAMERA_TOPIC, self._img, be)
        self.create_subscription(Odometry, ODOM_TOPIC, self._odom, be)
        self.create_subscription(String, "vla/command", self._cmd, qos)
        self.pub = self.create_publisher(MotionCommand, CONTROL_TOPIC, qos)
        self.create_timer(0.05, self._pub)
        self.get_logger().info(f"VLA-WP drive ready. GAIN={GAIN} Ld={LOOKAHEAD} CRUISE={CRUISE}")

    def _img(self, msg):
        self.latest = msg
        if not self.inferring:
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        with self.lock:
            if self.inferring:
                return
            self.inferring = True
        try:
            if self.bridge is None:
                from cv_bridge import CvBridge; self.bridge = CvBridge()
            bgr = self.bridge.imgmsg_to_cv2(self.latest, "bgr8")
            t0 = time.time()
            feat = extract_feature(self.vis, self.proc, bgr)[None]
            lane_t = torch.tensor([self.lane], dtype=torch.long, device="cuda:0")
            with torch.inference_mode():
                out = self.head(feat, lane_t)[0].cpu().numpy() * self.wp_scale
            self.fps = 0.9*self.fps + 0.1*(1.0/max(1e-3, time.time()-t0))
            pts = [(out[2*k], out[2*k+1]) for k in range(self.wp_n)]
            target = next(((ex, ey) for ex, ey in pts if ex >= LOOKAHEAD), pts[-1])
            ex, ey = target
            he = math.atan2(ey, max(0.3, ex))
            raw = -he * GAIN
            self.steer_f = STEER_EMA*raw + (1-STEER_EMA)*self.steer_f
            st = int(max(-7, min(7, round(self.steer_f))))
            with self.lock:
                self.steering = st
                self.speed = CRUISE if abs(st) < 3 else CRUISE_TURN
        finally:
            self.inferring = False

    def _parse_laps(self, t):
        if any(k in t for k in ("계속", "무한", "forever", "endless")):
            return 0
        m = re.search(r"(\d+)\s*바퀴", t) or re.search(r"(\d+)\s*lap", t)
        if m:
            return int(m.group(1))
        for k, v in KOR_NUM.items():
            if k+"바퀴" in t or k+" 바퀴" in t:
                return v
        return 1

    def _cmd(self, msg):
        t = msg.data.lower()
        if any(k in t for k in ("2차선", "이차선", "outer", "lane2", "lane 2", "lane two", "second lane")):
            self.lane = 1
        elif any(k in t for k in ("1차선", "일차선", "inner", "lane1", "lane 1", "lane one", "first lane")):
            self.lane = 0
        if any(k in t for k in ("멈춰", "정지", "stop")):
            with self.lock:
                self.active = False; self.target_laps = 0; self._last = None
            self.get_logger().info("[vla] 정지"); return
        laps = self._parse_laps(t)
        key = (self.lane, laps)
        if self.active and getattr(self, "_last", None) == key:
            return
        with self.lock:
            self._last = key
            self.active = True; self.target_laps = laps
            self.laps_done = 0; self.start_idx = None; self.visited = set()
        self.get_logger().info(f"[vla] lane={self.lane} 시작 ({'무한' if laps==0 else str(laps)+'바퀴'}) ~{self.fps:.0f}FPS")

    def _odom(self, msg):
        p = msg.pose.pose.position
        with self.lock:
            if not self.active or self.target_laps == 0:
                return
            cl = self.cl[self.lane]; N = len(cl)
            i = min(range(N), key=lambda k: (cl[k][0]-p.x)**2 + (cl[k][1]-p.y)**2)
            if self.start_idx is None:
                self.start_idx = i
            self.visited.add(i // 20)
            d = abs(i-self.start_idx); d = min(d, N-d)
            if len(self.visited) >= 34 and d < 12:
                self.laps_done += 1
                self.get_logger().info(f"🔁 {self.laps_done}/{self.target_laps}바퀴 (~{self.fps:.0f}FPS)")
                self.visited = set(); self.start_idx = i
                if self.laps_done >= self.target_laps:
                    self.active = False
                    self.get_logger().info(f"✅ {self.target_laps}바퀴 완료 — 정지")

    def _pub(self):
        with self.lock:
            st, sp = (self.steering, self.speed) if self.active else (0, 0)
        m = MotionCommand(); m.steering = int(st); m.left_speed = int(sp); m.right_speed = int(sp)
        self.pub.publish(m)


def main():
    rclpy.init(); node = VLADriveNode()
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

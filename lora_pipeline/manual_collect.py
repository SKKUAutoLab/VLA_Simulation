#!/usr/bin/env python3
"""
수동 주행 시연 녹화 — 사용자가 teleop로 운전하는 동안
(카메라 이미지 + 사용자 조향 + 차 pose)를 기록. 사용자의 '정답 라인'.
이미지→조향 모방학습 또는 (pose로) 경로 WP 라벨 둘 다 만들 수 있음.

함께 실행:
  ros2 launch simulation_pkg teleop_sim.launch.py   (또는 이미 sim 떠있으면 생략)
  python3 lora_pipeline/teleop_keyboard.py           # 사용자 운전
  python3 lora_pipeline/manual_collect.py --lane 0   # 본 녹화 (1차선 시연이면 0)

저장: manual_demos/images/man_*.jpg, manual_demos/labels.csv
      (fname, steering, speed, x, y, yaw, lane)
"""
import os, csv, math, time, argparse
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from interfaces_pkg.msg import MotionCommand
from cv_bridge import CvBridge

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "manual_demos")
IMG_DIR = os.path.join(OUT, "images")
SAVE_HZ = 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", type=int, default=0, help="이 시연이 몇 차선인지(0=1차선,1=2차선)")
    a, _ = ap.parse_known_args()
    os.makedirs(IMG_DIR, exist_ok=True)
    rclpy.init(); n = Node("manual_collect"); br = CvBridge()
    img = {"v": None}; cmd = {"st": 0, "sp": 0}; pose = {"x": None, "y": None, "yaw": 0.0}
    be = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
    rel = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE)
    n.create_subscription(Image, "camera/image_raw",
                          lambda m: img.__setitem__("v", br.imgmsg_to_cv2(m, "bgr8")), be)
    def ccb(m): cmd["st"], cmd["sp"] = m.steering, m.left_speed
    n.create_subscription(MotionCommand, "topic_control_signal", ccb, rel)
    def ocb(m):
        p = m.pose.pose.position; o = m.pose.pose.orientation
        pose["x"], pose["y"] = p.x, p.y
        pose["yaw"] = math.atan2(2*(o.w*o.z+o.x*o.y), 1-2*(o.y*o.y+o.z*o.z))
    n.create_subscription(Odometry, "/odom", ocb, be)

    csvp = os.path.join(OUT, "labels.csv")
    f = open(csvp, "a", newline=""); w = csv.writer(f)
    if os.path.getsize(csvp) == 0:
        w.writerow(["fname", "steering", "speed", "x", "y", "yaw", "lane"])
    n.get_logger().info(f"수동 녹화 시작 (lane {a.lane}). 운전하세요. 속도>0일 때만 기록. Ctrl-C로 종료.")
    last = 0.0; seq = 0; saved = 0
    try:
        while rclpy.ok():
            rclpy.spin_once(n, timeout_sec=0.02)
            now = time.time()
            if now - last < 1.0/SAVE_HZ or img["v"] is None or pose["x"] is None:
                continue
            if cmd["sp"] == 0:   # 정지 중에만 기록 안 함 (전진·후진 모두 기록)
                continue
            last = now
            fn = f"man_L{a.lane}_{int(time.time())}_{seq:06d}.jpg"
            cv2.imwrite(os.path.join(IMG_DIR, fn), img["v"])
            w.writerow([fn, cmd["st"], cmd["sp"], f"{pose['x']:.3f}", f"{pose['y']:.3f}",
                        f"{pose['yaw']:.4f}", a.lane]); f.flush()
            seq += 1; saved += 1
            if saved % 30 == 0:
                n.get_logger().info(f"기록 {saved}장 (st={cmd['st']} sp={cmd['sp']})")
    except KeyboardInterrupt:
        pass
    finally:
        f.close(); print(f"\n수동 녹화 종료: {saved}장 → {csvp}")
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

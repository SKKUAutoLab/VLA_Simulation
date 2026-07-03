#!/usr/bin/env python3
"""카메라 토픽 → mp4 녹화. 지정 시간 동안.
사용: python3 record_cam.py --topic top_camera/image_raw --out /tmp/a.mp4 --secs 30 --fps 20
"""
import argparse, time
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="top_camera/image_raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--secs", type=float, default=30)
    ap.add_argument("--fps", type=float, default=20)
    a, _ = ap.parse_known_args()
    rclpy.init()
    n = Node("record_cam")
    br = CvBridge()
    state = {"w": None, "frames": 0, "last": None}
    period = 1.0 / a.fps
    nxt = [0.0]

    def cb(m):
        img = br.imgmsg_to_cv2(m, "bgr8")
        state["last"] = img
        if state["w"] is None:
            h, w = img.shape[:2]
            state["w"] = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))

    n.create_subscription(Image, a.topic, cb,
                          QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT))
    t0 = time.time(); nxt[0] = t0
    while time.time() - t0 < a.secs and rclpy.ok():
        rclpy.spin_once(n, timeout_sec=0.02)
        now = time.time()
        if state["w"] is not None and state["last"] is not None and now >= nxt[0]:
            state["w"].write(state["last"]); state["frames"] += 1; nxt[0] += period
    if state["w"] is not None:
        state["w"].release()
    print(f"녹화 완료: {a.out}  {state['frames']} 프레임")
    rclpy.shutdown()


if __name__ == "__main__":
    main()

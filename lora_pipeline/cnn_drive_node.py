#!/usr/bin/env python3
"""
CNN 비전 주행 노드 — 카메라 이미지 → 조향(PilotNet). topic_control_signal 발행.
VLM 없이 순수 비전. 매우 빠름(수 ms) → 카메라 레이트로 제어.

실행:
    ros2 launch simulation_pkg teleop_sim.launch.py
    python3 lora_pipeline/cnn_drive_node.py
"""
import os
import cv2
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy,
                       QoSDurabilityPolicy)
from sensor_msgs.msg import Image
from interfaces_pkg.msg import MotionCommand
from train_cnn import PilotNet, preprocess

HERE = os.path.dirname(__file__)
MODEL = os.path.join(HERE, "cnn_model.pt")
CAMERA_TOPIC = "camera/image_raw"
CONTROL_TOPIC = "topic_control_signal"
CRUISE = 30


class CNNDriveNode(Node):
    def __init__(self):
        super().__init__("cnn_drive_node")
        ckpt = torch.load(MODEL, map_location="cuda:0")
        self.net = PilotNet().to("cuda:0")
        self.net(torch.zeros(1, 3, ckpt["in_h"], ckpt["in_w"], device="cuda:0"))
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        self.smax = ckpt["steer_max"]
        self.bridge = None
        self.steering = 0
        qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         durability=QoSDurabilityPolicy.VOLATILE, depth=1)
        self.create_subscription(
            Image, CAMERA_TOPIC, self._img,
            QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                       history=QoSHistoryPolicy.KEEP_LAST,
                       durability=QoSDurabilityPolicy.VOLATILE))
        self.pub = self.create_publisher(MotionCommand, CONTROL_TOPIC, qos)
        self.create_timer(0.05, self._pub)
        self.get_logger().info("CNN drive ready.")

    def _img(self, msg):
        if self.bridge is None:
            from cv_bridge import CvBridge
            self.bridge = CvBridge()
        bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        x = preprocess(bgr).transpose(2, 0, 1)[None]
        with torch.inference_mode():
            out = self.net(torch.from_numpy(x).to("cuda:0")).item()
        self.steering = int(max(-7, min(7, round(out * self.smax))))

    def _pub(self):
        m = MotionCommand()
        m.steering = int(self.steering)
        m.left_speed = CRUISE
        m.right_speed = CRUISE
        self.pub.publish(m)


def main():
    rclpy.init()
    node = CNNDriveNode()
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

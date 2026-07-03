#!/usr/bin/env python3
"""
Stage 4 — LoRA 비전 주행 노드
============================
학습된 LoRA 어댑터(lora_pipeline/adapter)를 base Qwen3-VL-2B에 얹어,
카메라만 보고 topic_control_signal(MotionCommand)을 발행한다.
bearing 힌트 없음 = 순수 비전. (build_dataset.PROMPT 와 동일 프롬프트 필수)

실행:
    ros2 launch simulation_pkg teleop_sim.launch.py    # 차+카메라+sender
    python3 lora_pipeline/lora_drive_node.py
"""
import os, re, time, threading
import cv2
import numpy as np
from PIL import Image as PILImage
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy,
                       QoSDurabilityPolicy)
from sensor_msgs.msg import Image
from interfaces_pkg.msg import MotionCommand
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from peft import PeftModel

HERE      = os.path.dirname(__file__)
ADAPTER   = os.path.join(HERE, "adapter")
BASE      = "Qwen/Qwen3-VL-2B-Instruct"
CAMERA_TOPIC  = "camera/image_raw"
CONTROL_TOPIC = "topic_control_signal"
INPUT_W, INPUT_H = 320, 240
PIXELS    = INPUT_W * INPUT_H
MAX_SPEED = 55
PUBLISH_HZ = 10.0

# build_dataset.PROMPT 과 반드시 동일
PROMPT = (
    "Drive a car using the front camera. Follow the gray road, stay between the "
    "lane lines. Reply ONE line only: D <st> <sp>  st=-7..7 (NEGATIVE=left, "
    "POSITIVE=right), sp=0..100; S=stop. Ex: D -3 40"
)


def parse(resp):
    s = resp.strip().lstrip("`*: ")
    up = s.upper()
    if up[:1] == "S" and not re.match(r'S?\s*-?\d', up):
        return 0, 0
    if up[:1] == "D":
        s = s[1:]
    nums = re.findall(r'-?\d+', s)
    if not nums:
        return 0, 0
    st = max(-7, min(7, int(nums[0])))
    sp = min(MAX_SPEED, int(nums[1])) if len(nums) > 1 else 0
    return st, sp


class LoRADriveNode(Node):
    def __init__(self):
        super().__init__("lora_drive_node")
        self.get_logger().info(f"Loading base {BASE} + adapter {ADAPTER} ...")
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            BASE, torch_dtype=torch.bfloat16, device_map="cuda:0",
            attn_implementation="sdpa")
        self.model = PeftModel.from_pretrained(base, ADAPTER).eval()
        self.proc = Qwen3VLProcessor.from_pretrained(
            BASE, min_pixels=PIXELS, max_pixels=PIXELS)
        self.bridge = None
        self.latest = None
        self.steering = 0
        self.speed = 0
        self.infer_lock = threading.Lock()
        self.inferring = False
        self._warmup()

        qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         durability=QoSDurabilityPolicy.VOLATILE, depth=1)
        self.create_subscription(
            Image, CAMERA_TOPIC, self._img,
            QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                       history=QoSHistoryPolicy.KEEP_LAST,
                       durability=QoSDurabilityPolicy.VOLATILE))
        self.pub = self.create_publisher(MotionCommand, CONTROL_TOPIC, qos)
        self.create_timer(1.0 / PUBLISH_HZ, self._pub_cb)
        self.get_logger().info("LoRA drive ready.")

    def _warmup(self):
        dummy = PILImage.fromarray(np.zeros((INPUT_H, INPUT_W, 3), np.uint8))
        self._infer(dummy)

    def _img(self, msg):
        self.latest = msg
        if not self.inferring:
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        with self.infer_lock:
            if self.inferring:
                return
            self.inferring = True
        try:
            if self.latest is None:
                return
            if self.bridge is None:
                from cv_bridge import CvBridge
                self.bridge = CvBridge()
            cv_img = self.bridge.imgmsg_to_cv2(self.latest, "bgr8")
            cv_img = cv2.resize(cv_img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
            pil = PILImage.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            resp = self._infer(pil)
            st, sp = parse(resp)
            self.steering, self.speed = st, sp
            self.get_logger().info(f"[vlm] {resp!r} → st={st} sp={sp}")
        finally:
            self.inferring = False

    def _infer(self, pil):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": pil}, {"type": "text", "text": PROMPT}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.proc(text=[text], images=[pil], return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            out = self.model.generate(**inp, max_new_tokens=10, do_sample=False, use_cache=True)
        new = out[0][inp.input_ids.shape[1]:]
        return self.proc.decode(new, skip_special_tokens=True).strip()

    def _pub_cb(self):
        m = MotionCommand()
        m.steering = int(self.steering)
        m.left_speed = int(self.speed)
        m.right_speed = int(self.speed)
        self.pub.publish(m)


def main():
    rclpy.init()
    node = LoRADriveNode()
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

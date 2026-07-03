#!/usr/bin/env python3
"""
Qwen3-VL-2B 기반 자율주행 제어 노드 (최적화 버전)

속도 개선 항목:
  1. max_new_tokens 32 → 16  (출력 JSON이 최대 ~15 토큰)
  2. attn_implementation="sdpa"  (PyTorch 내장 Scaled Dot-Product Attention)
  3. 이미지 해상도 축소: 640×480 → 320×240 (visual token 300 → ~70, ~4× 감소)
  4. processor min_pixels/max_pixels = 76800 으로 고정
  5. torch.compile(mode="reduce-overhead")  선택적 활성화
  6. use_cache=True 명시
  7. CUDA warmup
"""

import threading
import time
import json
import re
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
)
from sensor_msgs.msg import Image
from interfaces_pkg.msg import MotionCommand
from cv_bridge import CvBridge

import cv2
import numpy as np
from PIL import Image as PILImage

import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor


MODEL_NAME   = "Qwen/Qwen3-VL-2B-Instruct"
CAMERA_TOPIC = "camera/image_raw"
CONTROL_TOPIC = "topic_control_signal"
PUBLISH_HZ   = 10.0

# ──────────────────────────────────────────────
# 이미지 해상도
#   640×480 (307,200px, ~300 visual tokens): 정확도 높음 ← 기본값
#   320×240  (76,800px,  ~70 visual tokens): 커브+인접차량 오판 위험
# eval_resolution.py 평가 결과: 640×480 유지 권장
#   (320×240 속도 이득 1.13×에 불과, 커브 구간 steering ±5 오차 발생)
# ──────────────────────────────────────────────
INPUT_W = 640
INPUT_H = 480
# Qwen3-VL patch=28, merge=2 → 토큰당 56×56px
PROCESSOR_PIXELS = INPUT_W * INPUT_H   # 307,200

# max_new_tokens: {"steering": -7, "speed": 100} ≈ 15 tokens
MAX_NEW_TOKENS = 16

# torch.compile 사용 여부 (첫 추론 ~30s 컴파일 지연, 이후 ~1.5× 속도 향상)
# 환경변수 QWEN_COMPILE=1 로 활성화
USE_COMPILE = os.environ.get("QWEN_COMPILE", "0") == "1"

# /no_think 으로 thinking 비활성화 → 빠른 응답 + JSON만 출력
DRIVE_PROMPT = """/no_think
You are controlling an autonomous vehicle in a Gazebo simulation. \
Analyze the front camera image and output a single JSON driving command.

Output ONLY this JSON format, no explanation:
{"steering": <int -7 to 7>, "speed": <int 0 to 100>}

Steering: -7=hard left, 0=straight, +7=hard right
Speed: 0=stop, 100=full speed

Guidelines:
- Straight clear road → speed 80, steering 0
- Curve left → negative steering (-3 to -7), speed 50-70
- Curve right → positive steering (+3 to +7), speed 50-70
- Red traffic light or obstacle close ahead → speed 0, steering 0
- Yellow light → speed 20, steering 0"""


class QwenVLDriverNode(Node):
    def __init__(self):
        super().__init__('qwen_vl_driver_node')

        self.cv_bridge  = CvBridge()
        self.latest_image: Image | None = None
        self.is_inferring = False
        self.lock = threading.Lock()
        self.steering = 0
        self.speed    = 0

        # 추론 시간 통계
        self._infer_times: list[float] = []

        qos = QoSProfile(
            reliability  = QoSReliabilityPolicy.RELIABLE,
            history      = QoSHistoryPolicy.KEEP_LAST,
            durability   = QoSDurabilityPolicy.VOLATILE,
            depth        = 1,
        )

        # ── 모델 로드 ────────────────────────────────────────────────────
        self.get_logger().info(
            f"Loading {MODEL_NAME}  "
            f"(sdpa, {INPUT_W}×{INPUT_H}, max_new_tokens={MAX_NEW_TOKENS}, "
            f"compile={USE_COMPILE}) ..."
        )

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",   # ← SDPA: flash_attn 없이도 최적화
        )
        self.model.eval()

        # min/max pixels 고정 → 이미지 패치 수를 정확히 제어
        self.processor = Qwen3VLProcessor.from_pretrained(
            MODEL_NAME,
            min_pixels=PROCESSOR_PIXELS,
            max_pixels=PROCESSOR_PIXELS,
        )

        # ── torch.compile (선택) ─────────────────────────────────────────
        if USE_COMPILE:
            self.get_logger().info(
                "torch.compile(reduce-overhead) 시작 ... "
                "(첫 추론까지 ~30 s 컴파일 지연 발생)"
            )
            self.model = torch.compile(self.model, mode="reduce-overhead")

        # ── CUDA warmup ──────────────────────────────────────────────────
        self._warmup()

        self.get_logger().info(
            f"Qwen3-VL-2B ready. "
            f"GPU VRAM: {torch.cuda.memory_allocated() // 1024**2} MB"
        )

        self.sub   = self.create_subscription(Image, CAMERA_TOPIC, self._image_cb, qos)
        self.pub   = self.create_publisher(MotionCommand, CONTROL_TOPIC, qos)
        self.timer = self.create_timer(1.0 / PUBLISH_HZ, self._publish_cb)

    # ──────────────────────────────────────────────────────────────────────
    # Warmup
    # ──────────────────────────────────────────────────────────────────────
    def _warmup(self):
        """CUDA 그래프/커널 초기화 → 첫 추론 지연 제거."""
        try:
            dummy = PILImage.fromarray(
                np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)
            )
            inputs = self._build_inputs(dummy)
            with torch.inference_mode():
                self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                )
            self.get_logger().info("CUDA warmup done.")
        except Exception as e:
            self.get_logger().warn(f"Warmup skipped: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # 입력 빌드 (공통)
    # ──────────────────────────────────────────────────────────────────────
    def _build_inputs(self, pil_img: PILImage.Image) -> dict:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image",  "image": pil_img},
                {"type": "text",   "text": DRIVE_PROMPT},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.processor(
            text=[text],
            images=[pil_img],
            videos=None,
            padding=True,
            return_tensors="pt",
        ).to("cuda:0")

    # ──────────────────────────────────────────────────────────────────────
    # ROS 콜백
    # ──────────────────────────────────────────────────────────────────────
    def _image_cb(self, msg: Image):
        with self.lock:
            self.latest_image = msg
        if not self.is_inferring:
            threading.Thread(target=self._run_inference, daemon=True).start()

    def _run_inference(self):
        self.is_inferring = True
        try:
            with self.lock:
                if self.latest_image is None:
                    return
                msg = self.latest_image
                self.latest_image = None

            # BGR → RGB → resize → PIL
            cv_img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv_img = cv2.resize(cv_img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
            pil_img = PILImage.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

            inputs = self._build_inputs(pil_img)

            t0 = time.monotonic()
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                )
            elapsed = time.monotonic() - t0

            # 추론 시간 통계 (최근 10회 평균)
            self._infer_times.append(elapsed)
            if len(self._infer_times) > 10:
                self._infer_times.pop(0)
            avg = sum(self._infer_times) / len(self._infer_times)

            new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
            response = self.processor.decode(
                new_tokens, skip_special_tokens=True
            ).strip()

            self._parse_and_apply(response, elapsed, avg)

        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
        finally:
            self.is_inferring = False

    def _parse_and_apply(self, response: str, elapsed: float, avg: float):
        match = re.search(r'\{[^}]+\}', response)
        if not match:
            self.get_logger().warn(f"No JSON: '{response}'")
            return
        try:
            data = json.loads(match.group())
            self.steering = max(-7, min(7,   int(data["steering"])))
            self.speed    = max(0,  min(100, int(data["speed"])))
            self.get_logger().info(
                f"[{elapsed:.2f}s | avg {avg:.2f}s | {1/avg:.2f} FPS] "
                f"steering={self.steering}, speed={self.speed}"
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.get_logger().warn(f"Parse error ({e}): '{response}'")

    def _publish_cb(self):
        msg = MotionCommand()
        msg.steering    = self.steering
        msg.left_speed  = self.speed
        msg.right_speed = self.speed
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = QwenVLDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

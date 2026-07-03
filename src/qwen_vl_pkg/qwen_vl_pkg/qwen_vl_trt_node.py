#!/usr/bin/env python3
"""
Qwen3-VL-2B TensorRT 가속 노드

torch_tensorrt 를 사용해 LLM 디코더 레이어를 TRT 엔진으로 변환합니다.

사전 준비:
    pip install torch-tensorrt==2.10.0 \
        --extra-index-url https://download.pytorch.org/whl/cu128

실행 방법:
    # torch_tensorrt 사용 (기본)
    ros2 run qwen_vl_pkg qwen_vl_trt_node

    # 환경변수로 TRT 비활성화 (fallback to torch.compile/inductor)
    QWEN_TRT=0 ros2 run qwen_vl_pkg qwen_vl_trt_node

속도 향상 예상:
    baseline (torch.no_grad)   : ~4.0 s/frame  (~0.25 FPS)
    + SDPA + 320×240           : ~1.2 s/frame  (~0.8 FPS)
    + torch.compile inductor   : ~0.8 s/frame  (~1.2 FPS)
    + torch_tensorrt           : ~0.4 s/frame  (~2.5 FPS)  ← 목표
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


MODEL_NAME    = "Qwen/Qwen3-VL-2B-Instruct"
CAMERA_TOPIC  = "camera/image_raw"
CONTROL_TOPIC = "topic_control_signal"
PUBLISH_HZ    = 10.0

INPUT_W          = 320
INPUT_H          = 240
PROCESSOR_PIXELS = INPUT_W * INPUT_H   # 76,800
MAX_NEW_TOKENS   = 16

# 환경변수: QWEN_TRT=1(기본) → torch_tensorrt, =0 → torch.compile(inductor)
USE_TRT = os.environ.get("QWEN_TRT", "1") == "1"

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


def _compile_with_tensorrt(model: torch.nn.Module) -> torch.nn.Module:
    """
    torch_tensorrt.compile() 로 모델 컴파일.

    - dynamic=True : 가변 시퀀스 길이 지원 (VLM 필수)
    - precision=torch.bfloat16 : 정밀도 유지
    - enabled_precisions 에 torch.bfloat16 포함
    - truncate_long_and_double=True : int64 → int32 자동 변환
    """
    import torch_tensorrt  # noqa: F401

    compiled = torch.compile(
        model,
        backend="tensorrt",
        dynamic=True,
        options={
            "enabled_precisions": {torch.bfloat16},
            "truncate_long_and_double": True,
            "use_fast_partitioner": True,
            "optimization_level": 3,    # 0(빠른 빌드) ~ 5(최대 최적화)
            "cache_built_engines": True,
            "reuse_cached_engines": True,
        },
    )
    return compiled


def _compile_with_inductor(model: torch.nn.Module) -> torch.nn.Module:
    """
    torch_tensorrt 미설치 시 fallback: torch.compile(inductor).
    TRT보다 이득이 적지만 설치 없이 즉시 사용 가능.
    """
    return torch.compile(model, mode="reduce-overhead", dynamic=True)


class QwenVLTRTNode(Node):
    def __init__(self):
        super().__init__('qwen_vl_trt_node')

        self.cv_bridge = CvBridge()
        self.latest_image: Image | None = None
        self.is_inferring = False
        self.lock = threading.Lock()
        self.steering = 0
        self.speed    = 0
        self._infer_times: list[float] = []

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        # ── 모델 로드 ────────────────────────────────────────────────────
        self.get_logger().info(
            f"Loading {MODEL_NAME}  "
            f"(sdpa+{INPUT_W}×{INPUT_H}+max_new={MAX_NEW_TOKENS}) ..."
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
        )
        self.model.eval()

        self.processor = Qwen3VLProcessor.from_pretrained(
            MODEL_NAME,
            min_pixels=PROCESSOR_PIXELS,
            max_pixels=PROCESSOR_PIXELS,
        )

        # ── TRT / inductor 컴파일 ────────────────────────────────────────
        backend_used = "none"
        if USE_TRT:
            try:
                self.model = _compile_with_tensorrt(self.model)
                backend_used = "tensorrt"
                self.get_logger().info(
                    "torch_tensorrt 컴파일 시작 "
                    "(첫 추론에서 ~60 s TRT 엔진 빌드 진행)"
                )
            except ImportError:
                self.get_logger().warn(
                    "torch_tensorrt 미설치 → torch.compile(inductor) 로 fallback\n"
                    "  설치: pip install torch-tensorrt==2.10.0 "
                    "--extra-index-url https://download.pytorch.org/whl/cu128"
                )
                self.model = _compile_with_inductor(self.model)
                backend_used = "inductor"
        else:
            self.model = _compile_with_inductor(self.model)
            backend_used = "inductor"

        # ── Warmup (TRT 엔진 빌드 포함) ─────────────────────────────────
        self.get_logger().info(
            f"Warmup 시작 (backend={backend_used}) ... "
            "TRT 첫 실행 시 엔진 컴파일로 시간이 걸립니다."
        )
        self._warmup(n_runs=3 if backend_used == "tensorrt" else 2)
        self.get_logger().info(
            f"Ready! backend={backend_used}  "
            f"GPU: {torch.cuda.memory_allocated() // 1024**2} MB"
        )

        self.sub   = self.create_subscription(Image, CAMERA_TOPIC, self._image_cb, qos)
        self.pub   = self.create_publisher(MotionCommand, CONTROL_TOPIC, qos)
        self.timer = self.create_timer(1.0 / PUBLISH_HZ, self._publish_cb)

    # ──────────────────────────────────────────────────────────────────────
    def _warmup(self, n_runs: int = 2):
        dummy = PILImage.fromarray(
            np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)
        )
        inputs = self._build_inputs(dummy)
        for i in range(n_runs):
            t0 = time.monotonic()
            with torch.inference_mode():
                self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                )
            elapsed = time.monotonic() - t0
            self.get_logger().info(f"  warmup [{i+1}/{n_runs}]: {elapsed:.2f}s")

    # ──────────────────────────────────────────────────────────────────────
    def _build_inputs(self, pil_img: PILImage.Image) -> dict:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text",  "text":  DRIVE_PROMPT},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.processor(
            text=[text], images=[pil_img], videos=None,
            padding=True, return_tensors="pt",
        ).to("cuda:0")

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

            self._infer_times.append(elapsed)
            if len(self._infer_times) > 10:
                self._infer_times.pop(0)
            avg = sum(self._infer_times) / len(self._infer_times)

            new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
            response   = self.processor.decode(
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
    node = QwenVLTRTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

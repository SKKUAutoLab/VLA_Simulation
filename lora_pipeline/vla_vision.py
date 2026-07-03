#!/usr/bin/env python3
"""
Qwen3-VL-2B 비전 인코더 특징 추출 (공유) — 토큰생성 없이 1-pass(~10ms, 98FPS).
extract_feature(bgr) → (2048,) float32 mean-pooled 비전 특징.
학습/주행 노드가 동일 경로로 사용(분포 일치).
"""
import numpy as np
import torch
from PIL import Image as PILImage
import cv2

BASE = "Qwen/Qwen3-VL-2B-Instruct"
PIXELS = 320 * 240
FEAT_DIM = 2048
_DUMMY = None


def load_vision(device="cuda:0"):
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    m = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa").eval()
    proc = Qwen3VLProcessor.from_pretrained(BASE, min_pixels=PIXELS, max_pixels=PIXELS)
    return m.model.visual, proc


def _dummy(proc):
    global _DUMMY
    if _DUMMY is None:
        _DUMMY = proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": PILImage.new("RGB", (8, 8))},
                                          {"type": "text", "text": "x"}]}],
            tokenize=False, add_generation_prompt=True)
    return _DUMMY


@torch.inference_mode()
def extract_tokens(vis, proc, bgr, device="cuda:0"):
    """BGR → (Ntok, 2048) float32 (공간 토큰, 풀링 안 함)."""
    pil = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    inp = proc(text=[_dummy(proc)], images=[pil], return_tensors="pt").to(device)
    out = vis(inp["pixel_values"].to(torch.bfloat16), grid_thw=inp["image_grid_thw"])
    feat = out[0] if isinstance(out, (list, tuple)) else out   # (tokens, 2048)
    return feat.float()


@torch.inference_mode()
def extract_feature(vis, proc, bgr, device="cuda:0"):
    """BGR ndarray → (2048,) float32 mean-pool (구버전 호환)."""
    return extract_tokens(vis, proc, bgr, device).mean(dim=0)

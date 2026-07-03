#!/usr/bin/env python3
"""해상도(픽셀 예산) ↔ 비전 토큰 수 ↔ 비전 인코더 1-pass 추론속도(FPS) 측정.
해상도 올릴 때 실제로 FPS가 얼마나 떨어지는지 데이터로 확인."""
import time, glob, os
import cv2, numpy as np, torch
from PIL import Image as PILImage
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

BASE = "Qwen/Qwen3-VL-2B-Instruct"
img_path = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "dataset", "images", "*.jpg")))[0]
bgr = cv2.imread(img_path)
pil = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
print(f"테스트 이미지: {img_path} ({bgr.shape[1]}x{bgr.shape[0]})")

m = Qwen3VLForConditionalGeneration.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda:0",
                                                    attn_implementation="sdpa").eval()
vis = m.model.visual
budgets = [320*240, 448*336, 560*420, 640*480, 768*576, 896*672]
print(f"\n{'픽셀예산':>12} {'~해상도':>10} {'토큰':>6} {'추론ms':>8} {'FPS':>7}")
for P in budgets:
    proc = Qwen3VLProcessor.from_pretrained(BASE, min_pixels=P, max_pixels=P)
    t = proc.apply_chat_template([{"role": "user", "content": [{"type": "image", "image": pil},
                                  {"type": "text", "text": "x"}]}], tokenize=False, add_generation_prompt=True)
    inp = proc(text=[t], images=[pil], return_tensors="pt").to("cuda:0")
    g = inp["image_grid_thw"][0].tolist()
    ntok = (g[1]*g[2])//4
    ts = []
    for _ in range(12):
        t0 = time.time()
        with torch.inference_mode():
            vis(inp["pixel_values"].to(torch.bfloat16), grid_thw=inp["image_grid_thw"])
        torch.cuda.synchronize(); ts.append(time.time()-t0)
    ms = 1000*sum(ts[3:])/len(ts[3:])
    side = int(P**0.5)
    print(f"{P:>12} {f'~{side}x{side}':>10} {ntok:>6} {ms:>7.0f} {1000/ms:>7.1f}")
print("\n(WP헤드는 ~0.1ms 무시가능. ≥10FPS면 실시간 가능. 비동기청킹 쓰면 제어주기는 더 빠르게 분리.)")

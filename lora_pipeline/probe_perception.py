#!/usr/bin/env python3
"""
모델이 '차선/도로 방향'을 지각하는지 진단.
(1) BASE Qwen 자연어 질문: 도로가 좌/우/직진? → 순수 지각 능력
(2) LoRA 조향 예측: 좌/우/직선 이미지에 출력이 달라지나
val.jsonl 에서 gt 클래스별로 이미지를 골라 대조.
"""
import os, json, re
import cv2
from PIL import Image as PILImage
import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from peft import PeftModel

HERE = os.path.dirname(__file__)
BASE = "Qwen/Qwen3-VL-2B-Instruct"
ADAPTER = os.path.join(HERE, "adapter")
VAL = os.path.join(HERE, "dataset", "val.jsonl")
W, H = 320, 240
DRIVE_PROMPT = ("Drive a car using the front camera. Follow the gray road, stay between the "
    "lane lines. Reply ONE line only: D <st> <sp>  st=-7..7 (NEGATIVE=left, "
    "POSITIVE=right), sp=0..100; S=stop. Ex: D -3 40")
PERCEPT_PROMPT = ("Look at the road ahead in this driving camera image. Does the road go "
    "STRAIGHT, curve LEFT, or curve RIGHT? Answer with one word: STRAIGHT, LEFT, or RIGHT.")


def pick_by_gt(items, gt, k=2):
    return [it for it in items if it["steering"] == gt][:k]


def main():
    px = W * H
    proc = Qwen3VLProcessor.from_pretrained(BASE, min_pixels=px, max_pixels=px)
    items = [json.loads(l) for l in open(VAL)]
    # 클래스별 대표 이미지 (left=-6,-3 / straight=0 / right=3,6)
    samples = []
    for gt in (-6, -3, 0, 3, 6):
        samples += [(gt, it["image"]) for it in pick_by_gt(items, gt, 2)]

    def load(p):
        bgr = cv2.resize(cv2.imread(p), (W, H), interpolation=cv2.INTER_AREA)
        return PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def ask(model, pil, prompt, maxtok=8):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": pil}, {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[pil], return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=maxtok, do_sample=False, use_cache=True)
        return proc.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

    print("=== (1) BASE Qwen 지각 테스트: 도로 방향을 맞히나 ===")
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa").eval()
    expect = {-6:"LEFT", -3:"LEFT", 0:"STRAIGHT", 3:"RIGHT", 6:"RIGHT"}
    hit = 0
    for gt, p in samples:
        ans = ask(base, load(p), PERCEPT_PROMPT)
        ok = expect[gt].lower() in ans.lower()
        hit += ok
        print(f"  gt={gt:+d}(={expect[gt]:8}) → BASE: {ans!r} {'✓' if ok else '✗'}")
    print(f"  베이스 지각 정확도: {hit}/{len(samples)}")
    del base; torch.cuda.empty_cache()

    print("\n=== (2) LoRA 조향: 이미지따라 출력 변하나 ===")
    base2 = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa")
    lora = PeftModel.from_pretrained(base2, ADAPTER).eval()
    for gt, p in samples:
        ans = ask(lora, load(p), DRIVE_PROMPT, maxtok=10)
        print(f"  gt={gt:+d} → LoRA: {ans!r}")


if __name__ == "__main__":
    main()

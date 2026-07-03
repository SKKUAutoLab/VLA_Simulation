#!/usr/bin/env python3
"""
학습된 LoRA 어댑터를 val.jsonl 에서 평가 — 예측 steering vs expert 라벨.
base(어댑터 없음)와도 비교해 학습 효과 확인.
사용: python3 lora_pipeline/eval_lora.py [--n 80] [--base]
"""
import os, json, re, argparse
import cv2
from PIL import Image as PILImage
import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from peft import PeftModel

HERE = os.path.dirname(__file__)
ADAPTER = os.path.join(HERE, "adapter")
BASE = "Qwen/Qwen3-VL-2B-Instruct"
VAL = os.path.join(HERE, "dataset", "val.jsonl")
W, H = 320, 240
PROMPT = (
    "Drive a car using the front camera. Follow the gray road, stay between the "
    "lane lines. Reply ONE line only: D <st> <sp>  st=-7..7 (NEGATIVE=left, "
    "POSITIVE=right), sp=0..100; S=stop. Ex: D -3 40")


def parse_st(resp):
    s = resp.strip().lstrip("`*: ")
    if s[:1].upper() == "D":
        s = s[1:]
    m = re.findall(r'-?\d+', s)
    return int(m[0]) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--base", action="store_true", help="어댑터 없이 base만")
    args, _ = ap.parse_known_args()

    px = W * H
    proc = Qwen3VLProcessor.from_pretrained(BASE, min_pixels=px, max_pixels=px)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa")
    if not args.base:
        model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()

    items = [json.loads(l) for l in open(VAL)][:args.n]
    err = 0; n = 0; correct_sign = 0; preds = []
    for it in items:
        bgr = cv2.resize(cv2.imread(it["image"]), (W, H), interpolation=cv2.INTER_AREA)
        pil = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": pil}, {"type": "text", "text": PROMPT}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[pil], return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=10, do_sample=False, use_cache=True)
        resp = proc.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
        pred = parse_st(resp)
        gt = it["steering"]
        if pred is None:
            preds.append((gt, None, resp)); continue
        err += abs(pred - gt); n += 1
        if (pred > 0) == (gt > 0) or (pred == 0 and gt == 0):
            correct_sign += 1
        preds.append((gt, pred, resp))

    print(f"{'BASE' if args.base else 'LoRA'} | val {len(items)}장  파싱성공 {n}")
    if n:
        print(f"  steering MAE {err/n:.2f}  방향일치 {100*correct_sign/n:.0f}%")
    print("  샘플(gt,pred,raw):")
    for gt, pred, raw in preds[:15]:
        print(f"    gt={gt:+d} pred={pred} {raw!r}")


if __name__ == "__main__":
    main()

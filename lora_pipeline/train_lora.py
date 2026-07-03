#!/usr/bin/env python3
"""
Stage 3 — Qwen3-VL-2B LoRA 파인튜닝 (비전 주행)
=============================================
train.jsonl(image, prompt, target) → 언어모델 LoRA 학습. 비전 타워는 동결.
라벨은 프롬프트 구간을 -100으로 마스킹, 타깃("D <st> <sp>") 토큰만 supervise.

사용:
    python3 lora_pipeline/train_lora.py --epochs 3 --lr 1e-4
출력:
    lora_pipeline/adapter/   (PEFT 어댑터 — 추론 노드에서 로드)
"""
import os, json, argparse, math
import cv2
from PIL import Image as PILImage
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from peft import LoraConfig, get_peft_model

HERE      = os.path.dirname(__file__)
DATA      = os.path.join(HERE, "dataset")
ADAPTER   = os.path.join(HERE, "adapter")
MODEL     = "Qwen/Qwen3-VL-2B-Instruct"
INPUT_W, INPUT_H = 320, 240          # 추론 노드와 동일 해상도
PIXELS    = INPUT_W * INPUT_H


class DriveDataset(Dataset):
    def __init__(self, jsonl):
        self.items = [json.loads(l) for l in open(jsonl)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def build_labeled_inputs(proc, item):
    """full(프롬프트+정답) 토큰 - 프롬프트 길이만큼 -100 마스킹."""
    bgr = cv2.resize(cv2.imread(item["image"]), (INPUT_W, INPUT_H),
                     interpolation=cv2.INTER_AREA)
    pil = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    user = {"role": "user", "content": [
        {"type": "image", "image": pil}, {"type": "text", "text": item["prompt"]}]}
    asst = {"role": "assistant", "content": [{"type": "text", "text": item["target"]}]}

    text_full   = proc.apply_chat_template([user, asst], tokenize=False,
                                           add_generation_prompt=False)
    text_prompt = proc.apply_chat_template([user], tokenize=False,
                                           add_generation_prompt=True)
    full = proc(text=[text_full], images=[pil], return_tensors="pt")
    # 같은 이미지라 vision 토큰 수 동일 → 프롬프트 길이로 prefix 마스킹
    plen = proc(text=[text_prompt], images=[pil],
                return_tensors="pt").input_ids.shape[1]
    labels = full.input_ids.clone()
    labels[:, :plen] = -100
    full["labels"] = labels
    return {k: v[0] for k, v in full.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8, help="gradient accumulation")
    ap.add_argument("--r", type=int, default=16)
    args, _ = ap.parse_known_args()

    proc = Qwen3VLProcessor.from_pretrained(MODEL, min_pixels=PIXELS, max_pixels=PIXELS)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa")
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.r, lora_alpha=args.r * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        # 언어층(q/k/v/o,gate/up/down) + 비전→언어 커넥터/비전MLP(linear_fc1/2)
        # 커넥터까지 학습해야 이미지 특징이 조향에 반영됨(상수붕괴 방지)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "linear_fc1", "linear_fc2"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = DriveDataset(os.path.join(DATA, "train.jsonl"))
    dl = DataLoader(ds, batch_size=1, shuffle=True,
                    collate_fn=lambda b: build_labeled_inputs(proc, b[0]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total_steps = math.ceil(len(dl) / args.accum) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, total_steps), pct_start=0.1)

    model.train()
    step = 0
    for ep in range(args.epochs):
        running = 0.0
        opt.zero_grad()
        for i, batch in enumerate(dl):
            batch = {k: v.unsqueeze(0).to("cuda:0") for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.accum
            loss.backward()
            running += out.loss.item()
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 10 == 0:
                    print(f"ep{ep} step{step}/{total_steps} "
                          f"loss={running/(i+1):.4f} lr={sched.get_last_lr()[0]:.2e}")
        print(f"== epoch {ep} 완료  avg_loss={running/len(dl):.4f}")

    os.makedirs(ADAPTER, exist_ok=True)
    model.save_pretrained(ADAPTER)
    proc.save_pretrained(ADAPTER)
    print(f"\n✓ 어댑터 저장 → {ADAPTER}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
E1-ablation: 고해상 + spatial attn-pool, 비전 FROZEN (LoRA 없음).
"해상도/공간정보 부족"이 1차선 실패 원인인지 싸게 분리 검증.
고해상 토큰(frozen) 캐싱 → AttnHead만 학습(빠름).
저장: vla_hifrozen_head.pt (meta: pixels, ntok, grid)
"""
import os, csv, glob, random, time
import cv2, numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image as PILImage
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from train_wp import WP_N, WP_SCALE
from train_vla_hires import AttnHead

HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "dataset")
BASE = "Qwen/Qwen3-VL-2B-Instruct"
PIXELS = int(os.environ.get("VLA_PIXELS", "235200"))
TOKCACHE = os.path.join(DATA, f"vla_tok_{PIXELS}.pt")
OUT = os.path.join(HERE, "vla_hifrozen_head.pt")
FEAT = 2048; SEED = 0; EPOCHS = 60


def load_rows():
    rows = []
    for cf in sorted(glob.glob(os.path.join(DATA, "labels_wpL*.csv"))):
        for r in csv.DictReader(open(cf)):
            ip = os.path.join(DATA, "images", r["fname"])
            if not os.path.exists(ip):
                continue
            wps = [float(r[f"{c}{k}"]) for k in range(WP_N) for c in ("ex", "ey")]
            rows.append((ip, wps, int(r["lane"])))
    return rows


@torch.inference_mode()
def build_tok(paths):
    cache = torch.load(TOKCACHE) if os.path.exists(TOKCACHE) else {}
    miss = [p for p in paths if p not in cache]
    grid = None
    if miss:
        print(f"고해상 토큰 추출 {len(miss)}개 @ {PIXELS}px...", flush=True)
        m = Qwen3VLForConditionalGeneration.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda:0",
                                                            attn_implementation="sdpa").eval()
        vis = m.model.visual
        proc = Qwen3VLProcessor.from_pretrained(BASE, min_pixels=PIXELS, max_pixels=PIXELS)
        dummy = proc.apply_chat_template([{"role": "user", "content": [
            {"type": "image", "image": PILImage.new("RGB", (8, 8))}, {"type": "text", "text": "x"}]}],
            tokenize=False, add_generation_prompt=True)
        for i, p in enumerate(miss):
            bgr = cv2.imread(p)
            if bgr is None:
                continue
            pil = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            inp = proc(text=[dummy], images=[pil], return_tensors="pt").to("cuda:0")
            out = vis(inp["pixel_values"].to(torch.bfloat16), grid_thw=inp["image_grid_thw"])
            feat = out[0] if isinstance(out, (list, tuple)) else out
            cache[p] = feat.half().cpu()
            if (i+1) % 500 == 0:
                print(f"  {i+1}/{len(miss)}", flush=True)
        torch.save(cache, TOKCACHE); del m, vis; torch.cuda.empty_cache()
        print(f"토큰캐시 저장 {len(cache)}", flush=True)
    else:
        print(f"토큰캐시 사용 {len(cache)}", flush=True)
    return cache


class TokSet(Dataset):
    def __init__(self, rows, cache): self.rows = rows; self.cache = cache
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        p, wps, lane = self.rows[i]
        return (self.cache[p].float(), torch.tensor(lane, dtype=torch.long),
                torch.tensor([v/WP_SCALE for v in wps], dtype=torch.float32))


def main():
    torch.manual_seed(SEED); dev = "cuda:0"; nout = 2*WP_N
    rows = load_rows()
    cache = build_tok(sorted({p for p, _, _ in rows}))
    rows = [r for r in rows if r[0] in cache]
    ntok = next(iter(cache.values())).shape[0]
    print(f"NTOK={ntok}", flush=True)
    random.Random(SEED).shuffle(rows)
    nv = max(1, int(len(rows)*0.1)); val, tr = rows[:nv], rows[nv:]
    print(f"train {len(tr)} / val {len(val)}", flush=True)
    tl = DataLoader(TokSet(tr, cache), batch_size=64, shuffle=True, num_workers=2)
    vl = DataLoader(TokSet(val, cache), batch_size=64, num_workers=2)
    net = AttnHead(nout).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=7e-4, weight_decay=1e-5)
    lossf = nn.MSELoss(); best = 1e9
    for ep in range(EPOCHS):
        net.train(); tre = 0
        for f, lane, y in tl:
            f, lane, y = f.to(dev), lane.to(dev), y.to(dev)
            opt.zero_grad(); l = lossf(net(f, lane), y); l.backward(); opt.step(); tre += l.item()*len(f)
        net.eval(); mae = {0: 0, 1: 0}; cnt = {0: 0, 1: 0}
        with torch.no_grad():
            for f, lane, y in vl:
                f, lane, y = f.to(dev), lane.to(dev), y.to(dev); p = net(f, lane)
                for li in (0, 1):
                    msk = lane == li
                    if msk.any(): mae[li] += (p[msk]-y[msk]).abs().sum().item()*WP_SCALE; cnt[li] += msk.sum().item()*nout
        m0 = mae[0]/max(1, cnt[0]); m1 = mae[1]/max(1, cnt[1]); mm = (m0+m1)/2
        if mm < best:
            best = mm
            torch.save({"state_dict": net.state_dict(), "wp_n": WP_N, "wp_scale": WP_SCALE,
                        "nout": nout, "pixels": PIXELS, "ntok": ntok}, OUT)
        if ep % 5 == 0 or ep == EPOCHS-1:
            print(f"ep{ep} mse={tre/len(tr):.4f} val 1차선={m0:.2f} 2차선={m1:.2f} (best {best:.2f})", flush=True)
    print(f"\n✓ best wpMAE {best:.2f}m → {OUT}", flush=True)


if __name__ == "__main__":
    main()

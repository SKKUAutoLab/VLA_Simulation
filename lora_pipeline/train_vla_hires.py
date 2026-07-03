#!/usr/bin/env python3
"""
E1: 고해상 + spatial attention-pool + FiLM lane + 비전 LoRA (end-to-end).
가설: 1차선 급코너 실패 = 저해상(320x240)+mean-pool로 미세곡률 소실.
→ 해상도↑ + 공간토큰 보존(attn-pool) + 강한 lane조건부(FiLM) + 비전적응(LoRA)로 돌파.
저장: vla_hires_adapter/ + vla_hires_head.pt (meta: pixels, ntok)
"""
import os, csv, glob, random, time
import cv2, numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image as PILImage
from peft import LoraConfig, get_peft_model
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from train_wp import WP_N, WP_SCALE

HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "dataset")
BASE = "Qwen/Qwen3-VL-2B-Instruct"
PIXELS = int(os.environ.get("VLA_PIXELS", "235200"))   # 484^2, ~221토큰, ~55FPS
PXCACHE = os.path.join(DATA, f"vla_px_{PIXELS}.pt")
ADAPTER = os.path.join(HERE, "vla_hires_adapter")
HEAD = os.path.join(HERE, "vla_hires_head.pt")
FEAT = 2048; SEED = 0
EPOCHS = int(os.environ.get("EPOCHS", "24"))
BATCH = int(os.environ.get("BATCH", "8"))


class AttnHead(nn.Module):
    """spatial attention-pool(공간정보 보존) + FiLM lane조건부."""
    def __init__(self, nout, dim=FEAT, nq=8):
        super().__init__()
        self.q = nn.Parameter(torch.randn(nq, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.ln = nn.LayerNorm(dim)
        self.proj = nn.Linear(nq * dim, 512)
        self.film = nn.Embedding(2, 1024); nn.init.zeros_(self.film.weight)
        self.net = nn.Sequential(nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, nout))

    def forward(self, tokens, lane):        # tokens (B,T,dim)
        B = tokens.shape[0]
        t = self.ln(tokens)
        q = self.q.unsqueeze(0).expand(B, -1, -1)
        p, _ = self.attn(q, t, t)           # (B,nq,dim)
        h = self.proj(p.reshape(B, -1))     # (B,512)
        g, b = self.film(lane).chunk(2, dim=-1)
        h = h * (1 + g) + b
        return self.net(h)


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


def build_px(paths, proc):
    cache = torch.load(PXCACHE) if os.path.exists(PXCACHE) else {}
    miss = [p for p in paths if p not in cache]
    if miss:
        print(f"pixel_values 추출 {len(miss)}개 @ {PIXELS}px...", flush=True)
        dummy = proc.apply_chat_template([{"role": "user", "content": [
            {"type": "image", "image": PILImage.new("RGB", (8, 8))}, {"type": "text", "text": "x"}]}],
            tokenize=False, add_generation_prompt=True)
        npatch = None
        for i, p in enumerate(miss):
            bgr = cv2.imread(p)
            if bgr is None:
                continue
            pil = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            px = proc(text=[dummy], images=[pil], return_tensors="pt")["pixel_values"]
            if npatch is None:
                npatch = px.shape[0]
            if px.shape[0] != npatch:
                continue
            cache[p] = px.half()
            if (i+1) % 500 == 0:
                print(f"  {i+1}/{len(miss)}", flush=True)
        torch.save(cache, PXCACHE); print(f"px캐시 저장 {len(cache)} (npatch={npatch})", flush=True)
    else:
        print(f"px캐시 사용 {len(cache)}", flush=True)
    return cache


class PxSet(Dataset):
    def __init__(self, rows, cache): self.rows = rows; self.cache = cache
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        p, wps, lane = self.rows[i]
        return (self.cache[p], lane, torch.tensor([v/WP_SCALE for v in wps], dtype=torch.float32))


def main():
    torch.manual_seed(SEED); dev = "cuda:0"; nout = 2*WP_N
    m = Qwen3VLForConditionalGeneration.from_pretrained(BASE, dtype=torch.bfloat16, device_map=dev,
                                                        attn_implementation="sdpa").eval()
    vis = m.model.visual.float()
    proc = Qwen3VLProcessor.from_pretrained(BASE, min_pixels=PIXELS, max_pixels=PIXELS)
    rows = load_rows()
    cache = build_px(sorted({p for p, _, _ in rows}), proc)
    rows = [r for r in rows if r[0] in cache]
    # 토큰 수/그리드 파악
    sample_px = next(iter(cache.values()))
    npatch = sample_px.shape[0]
    # grid_thw: pixel_values는 (npatch,1536), 토큰=npatch//4 (merge2x2)
    NTOK = npatch // 4
    # grid 복원: proc로 한 번
    dummy = proc.apply_chat_template([{"role": "user", "content": [
        {"type": "image", "image": PILImage.new("RGB", (8, 8))}, {"type": "text", "text": "x"}]}],
        tokenize=False, add_generation_prompt=True)
    bgr0 = cv2.imread(rows[0][0]); pil0 = PILImage.fromarray(cv2.cvtColor(bgr0, cv2.COLOR_BGR2RGB))
    grid = proc(text=[dummy], images=[pil0], return_tensors="pt")["image_grid_thw"][0].tolist()
    print(f"npatch={npatch} NTOK={NTOK} grid={grid}", flush=True)

    random.Random(SEED).shuffle(rows)
    nv = max(1, int(len(rows)*0.1)); val, tr = rows[:nv], rows[nv:]
    print(f"train {len(tr)} / val {len(val)}", flush=True)

    cfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
                     target_modules=["qkv", "proj", "linear_fc1", "linear_fc2"])
    vis = get_peft_model(vis, cfg); vis.print_trainable_parameters()
    head = AttnHead(nout).float().to(dev)

    def collate(batch):
        px = torch.cat([b[0] for b in batch], 0)
        lane = torch.tensor([b[1] for b in batch], dtype=torch.long)
        y = torch.stack([b[2] for b in batch])
        gt = torch.tensor([grid]*len(batch), dtype=torch.long)
        return px, gt, lane, y

    def run(px, gt):
        out = vis(px.float().to(dev), grid_thw=gt.to(dev))
        feat = out[0] if isinstance(out, (list, tuple)) else out   # (B*NTOK, dim)
        return feat.view(-1, NTOK, FEAT)

    tl = DataLoader(PxSet(tr, cache), batch_size=BATCH, shuffle=True, num_workers=2, collate_fn=collate)
    vl = DataLoader(PxSet(val, cache), batch_size=BATCH, num_workers=2, collate_fn=collate)
    params = [p for p in vis.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.Adam(params, lr=3e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.MSELoss(); best = 1e9
    for ep in range(EPOCHS):
        vis.train(); head.train(); tre = 0; t0 = time.time()
        for px, gt, lane, y in tl:
            lane, y = lane.to(dev), y.to(dev)
            opt.zero_grad(); pred = head(run(px, gt), lane); l = lossf(pred, y)
            l.backward(); opt.step(); tre += l.item()*len(y)
        sched.step()
        vis.eval(); head.eval(); mae = {0: 0, 1: 0}; cnt = {0: 0, 1: 0}
        with torch.no_grad():
            for px, gt, lane, y in vl:
                lane, y = lane.to(dev), y.to(dev); p = head(run(px, gt), lane)
                for li in (0, 1):
                    msk = lane == li
                    if msk.any(): mae[li] += (p[msk]-y[msk]).abs().sum().item()*WP_SCALE; cnt[li] += msk.sum().item()*nout
        m0 = mae[0]/max(1, cnt[0]); m1 = mae[1]/max(1, cnt[1]); mm = (m0+m1)/2
        if mm < best:
            best = mm
            vis.save_pretrained(ADAPTER)
            torch.save({"state_dict": head.state_dict(), "wp_n": WP_N, "wp_scale": WP_SCALE,
                        "nout": nout, "pixels": PIXELS, "ntok": NTOK, "grid": grid}, HEAD)
        print(f"ep{ep} mse={tre/len(tr):.4f} val 1차선={m0:.2f} 2차선={m1:.2f} (best {best:.2f}) {time.time()-t0:.0f}s", flush=True)
    print(f"\n✓ best wpMAE {best:.2f}m → {ADAPTER} + {HEAD}", flush=True)


if __name__ == "__main__":
    main()

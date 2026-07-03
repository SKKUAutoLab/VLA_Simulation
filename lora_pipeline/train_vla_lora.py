#!/usr/bin/env python3
"""
순수 VLA WP 학습 (비전 인코더 LoRA, 비동결) — 동결특징 천장 돌파 시도.
Qwen 비전(LoRA) → mean-pool → WP 헤드, end-to-end. pixel_values 캐시(불변)로 가속.
저장: vla_lora_adapter/ (비전 LoRA) + vla_lora_head.pt
"""
import os, csv, glob, random, time
import cv2, numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image as PILImage
from peft import LoraConfig, get_peft_model
from train_wp import WP_N, WP_SCALE
from vla_vision import load_vision, FEAT_DIM, _dummy

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "dataset")
PXCACHE = os.path.join(DATA, "vla_px_cache.pt")
LANE_ONLY = os.environ.get("LANE_ONLY")   # "0"/"1" 이면 해당 차선만 학습(per-lane 어댑터)
_SUF = f"_l{LANE_ONLY}" if LANE_ONLY in ("0", "1") else ""
ADAPTER = os.path.join(HERE, f"vla_lora_adapter{_SUF}")
HEAD = os.path.join(HERE, f"vla_lora_head{_SUF}.pt")
NPATCH = 280; NTOK = 70
SEED = 0


class Head(nn.Module):
    """공간보존 헤드 — 토큰 7×10 그리드의 위치(특히 가로=횡방향 차선위치)를 유지.
    mean-pool은 차선의 좌우 치우침을 지워 직선 횡오차 복구 불가 → 토큰별 투영+위치임베딩 후
    전체 평탄화로 공간정보를 헤드에 그대로 전달. FiLM lane 조건부는 유지."""
    def __init__(self, nout, ntok=NTOK):
        super().__init__()
        self.tok = nn.Linear(FEAT_DIM, 64)               # 토큰별 차원축소(2048→64)
        self.pos = nn.Parameter(torch.zeros(1, ntok, 64))  # 학습 위치임베딩(슬롯=공간위치)
        self.proj = nn.Linear(ntok * 64, 512)            # 전 토큰 평탄화→공간배치 보존
        self.film = nn.Embedding(4, 1024)   # gamma(512)+beta(512). 0/1=차선유지, 2/3=차선변경
        nn.init.zeros_(self.film.weight)     # 시작은 항등(h*1+0)
        self.net = nn.Sequential(nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, nout))
    def forward(self, feat, lane):
        # feat: (B, ntok, FEAT_DIM) — 풀링하지 않은 공간 토큰
        h = self.tok(feat) + self.pos        # (B,ntok,64)
        h = self.proj(h.flatten(1))          # (B,512)
        g, b = self.film(lane).chunk(2, dim=-1)
        h = h * (1 + g) + b
        return self.net(h)


def load_rows():
    rows = []
    # 1) 텔레포트 복구 WP (dataset/images/<fname>)
    for cf in sorted(glob.glob(os.path.join(DATA, "labels_wpL*.csv"))):
        for r in csv.DictReader(open(cf)):
            ip = os.path.join(DATA, "images", r["fname"])
            if not os.path.exists(ip):
                continue
            lane = int(r["lane"])
            if LANE_ONLY in ("0", "1") and lane != int(LANE_ONLY):
                continue
            wps = [float(r[f"{c}{k}"]) for k in range(WP_N) for c in ("ex", "ey")]
            rows.append((ip, wps, lane))
    # 2) 수동 시연 WP (절대경로 path) — USE_MANUAL=1 일 때
    manp = os.path.join(HERE, "manual_wp_labels.csv")
    if os.environ.get("USE_MANUAL", "1") == "1" and os.path.exists(manp):
        n0 = len(rows)
        for r in csv.DictReader(open(manp)):
            ip = r["path"]
            if not os.path.exists(ip):
                continue
            lane = int(r["lane"])
            if LANE_ONLY in ("0", "1") and lane != int(LANE_ONLY):
                continue
            wps = [float(r[f"{c}{k}"]) for k in range(WP_N) for c in ("ex", "ey")]
            rows.append((ip, wps, lane))
        print(f"수동 WP {len(rows)-n0}개 추가")
    return rows


def build_px(paths, proc):
    cache = torch.load(PXCACHE) if os.path.exists(PXCACHE) else {}
    miss = [p for p in paths if p not in cache]
    if miss:
        print(f"pixel_values 추출 {len(miss)}개...")
        for i, p in enumerate(miss):
            bgr = cv2.imread(p)
            if bgr is None:
                continue
            pil = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            inp = proc(text=[_dummy(proc)], images=[pil], return_tensors="pt")
            px = inp["pixel_values"]
            if px.shape[0] != NPATCH:
                continue
            cache[p] = px.half()
            if (i+1) % 500 == 0:
                print(f"  {i+1}/{len(miss)}")
        torch.save(cache, PXCACHE); print(f"px캐시 저장 {len(cache)}")
    else:
        print(f"px캐시 사용 {len(cache)}")
    return cache


class PxSet(Dataset):
    def __init__(self, rows, cache): self.rows = rows; self.cache = cache
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        p, wps, lane = self.rows[i]
        return (self.cache[p], lane, torch.tensor([v/WP_SCALE for v in wps], dtype=torch.float32))


def collate(batch):
    px = torch.cat([b[0] for b in batch], 0)            # (B*280,1536)
    lane = torch.tensor([b[1] for b in batch], dtype=torch.long)
    y = torch.stack([b[2] for b in batch])
    B = len(batch)
    grid = torch.tensor([[1, 14, 20]]*B, dtype=torch.long)
    return px, grid, lane, y


def main():
    torch.manual_seed(SEED)
    dev = "cuda:0"; nout = 2*WP_N
    vis, proc = load_vision()
    rows = load_rows()
    cache = build_px(sorted({p for p, _, _ in rows}), proc)
    rows = [r for r in rows if r[0] in cache]
    random.Random(SEED).shuffle(rows)
    nv = max(1, int(len(rows)*0.1)); val, tr = rows[:nv], rows[nv:]
    print(f"train {len(tr)} / val {len(val)}")

    vis = vis.float()
    cfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
                     target_modules=["qkv", "proj", "linear_fc1", "linear_fc2"])
    vis = get_peft_model(vis, cfg)
    vis.print_trainable_parameters()
    head = Head(nout).float().to(dev)

    def run(px, grid):
        out = vis(px.float().to(dev), grid_thw=grid.to(dev))
        feat = out[0] if isinstance(out, (list, tuple)) else out   # (B*70,2048)
        return feat.view(-1, NTOK, FEAT_DIM)                       # (B,70,2048) 공간보존(풀링X)

    tl = DataLoader(PxSet(tr, cache), batch_size=16, shuffle=True, num_workers=2, collate_fn=collate)
    vl = DataLoader(PxSet(val, cache), batch_size=16, num_workers=2, collate_fn=collate)
    params = [p for p in vis.parameters() if p.requires_grad] + list(head.parameters())
    EPOCHS = int(os.environ.get("EPOCHS", "24"))
    opt = torch.optim.Adam(params, lr=3e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.MSELoss(); best = 1e9
    for ep in range(EPOCHS):
        vis.train(); head.train(); tre = 0; t0 = time.time()
        for px, grid, lane, y in tl:
            lane, y = lane.to(dev), y.to(dev)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):   # forward만 bf16(~2x), LoRA파라미터는 fp32
                pred = head(run(px, grid), lane); l = lossf(pred, y)
            l.backward(); opt.step(); tre += l.item()*len(y)
        vis.eval(); head.eval(); mae = {0: 0, 1: 0}; cnt = {0: 0, 1: 0}
        with torch.no_grad():
            for px, grid, lane, y in vl:
                lane, y = lane.to(dev), y.to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = head(run(px, grid), lane)
                p = p.float()
                for li in (0, 1):
                    m = lane == li
                    if m.any(): mae[li] += (p[m]-y[m]).abs().sum().item()*WP_SCALE; cnt[li] += m.sum().item()*nout
        m0 = mae[0]/max(1, cnt[0]); m1 = mae[1]/max(1, cnt[1])
        # 단일차선 학습 시 해당 차선 MAE로만 best 선택(빈 차선 0이 평균 왜곡 방지)
        mm = m0 if LANE_ONLY == "0" else (m1 if LANE_ONLY == "1" else (m0+m1)/2)
        if mm < best:
            best = mm
            vis.save_pretrained(ADAPTER)
            torch.save({"state_dict": head.state_dict(), "wp_n": WP_N, "wp_scale": WP_SCALE, "nout": nout}, HEAD)
        sched.step()
        print(f"ep{ep} mse={tre/len(tr):.4f} val 1차선={m0:.2f} 2차선={m1:.2f} (best {best:.2f}) {time.time()-t0:.0f}s")
    print(f"\n✓ best wpMAE {best:.2f}m → {ADAPTER} + {HEAD}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
순수 VLA 웨이포인트 학습 (B) — Qwen3-VL 비전특징(동결) + WP 회귀 헤드.
토큰조향 X. 1-pass 비전 → 헤드 → 미래 ego 웨이포인트. 기존 WP 라벨 재사용.
비전특징은 한 번 추출해 캐시(dataset/vla_feat_cache.pt). 저장: vla_wp_head.pt
"""
import os, csv, glob, random, time
import cv2, numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from train_wp import WP_N, WP_SCALE
from vla_vision import load_vision, extract_feature, FEAT_DIM

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "dataset")
CACHE = os.path.join(DATA, "vla_feat_cache.pt")
OUT = os.path.join(HERE, "vla_wp_head.pt")
SEED = 0


class Head(nn.Module):
    def __init__(self, nout):
        super().__init__()
        self.emb = nn.Embedding(2, 16)
        self.net = nn.Sequential(
            nn.Linear(FEAT_DIM + 16, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, nout))

    def forward(self, feat, lane):
        return self.net(torch.cat([feat, self.emb(lane)], dim=1))


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


def build_cache(paths):
    if os.path.exists(CACHE):
        cache = torch.load(CACHE)
        miss = [p for p in paths if p not in cache]
        if not miss:
            print(f"캐시 사용 {len(cache)}개"); return cache
    else:
        cache = {}; miss = list(paths)
    print(f"비전특징 추출 {len(miss)}개 (캐시 {len(cache)})...")
    vis, proc = load_vision()
    t0 = time.time()
    for i, p in enumerate(miss):
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        cache[p] = extract_feature(vis, proc, bgr).cpu()
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{len(miss)} ({(time.time()-t0)/(i+1)*1000:.0f}ms/장)")
    torch.save(cache, CACHE)
    del vis; torch.cuda.empty_cache()
    print(f"캐시 저장 {len(cache)}개 → {CACHE}")
    return cache


class FeatSet(Dataset):
    def __init__(self, rows, cache): self.rows = rows; self.cache = cache
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        p, wps, lane = self.rows[i]
        return (self.cache[p], torch.tensor(lane, dtype=torch.long),
                torch.tensor([v/WP_SCALE for v in wps], dtype=torch.float32))


def main():
    torch.manual_seed(SEED)
    rows = load_rows()
    cache = build_cache(sorted({p for p, _, _ in rows}))
    rows = [r for r in rows if r[0] in cache]
    random.Random(SEED).shuffle(rows)
    nv = max(1, int(len(rows)*0.1)); val, tr = rows[:nv], rows[nv:]
    print(f"train {len(tr)} / val {len(val)}")
    dev = "cuda:0"; nout = 2*WP_N
    tl = DataLoader(FeatSet(tr, cache), batch_size=128, shuffle=True, num_workers=2)
    vl = DataLoader(FeatSet(val, cache), batch_size=128, num_workers=2)
    net = Head(nout).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.MSELoss(); best = 1e9
    for ep in range(80):
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
            torch.save({"state_dict": net.state_dict(), "wp_n": WP_N, "wp_scale": WP_SCALE, "nout": nout}, OUT)
        if ep % 5 == 0 or ep == 79:
            print(f"ep{ep} mse={tre/len(tr):.4f} val_wpMAE 1차선={m0:.2f} 2차선={m1:.2f} (best {best:.2f})")
    print(f"\n✓ best wpMAE {best:.2f}m → {OUT}")


if __name__ == "__main__":
    main()

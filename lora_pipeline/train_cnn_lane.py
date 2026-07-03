#!/usr/bin/env python3
"""
차선조건부 CNN — (카메라 이미지 + 차선ID) → 조향.
lane 0 = 1차선(inner), lane 1 = 2차선(outer). CSV 출처로 차선 라벨 부여.
저장: lora_pipeline/cnn_lane_model.pt
"""
import os, csv, math, random, collections
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from train_cnn import preprocess, IN_W, IN_H, STEER_MAX   # 전처리 공유

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "dataset")
IMG_DIR = os.path.join(DATA, "images")
OUT = os.path.join(HERE, "cnn_lane_model.pt")
import glob
CAP = 400                  # (lane,조향)별 상한
SEED = 0


NUM_CLASSES = 15   # 조향 -7..7 (15 이산레벨) 분류


class LaneSet(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        path, st, lane = self.rows[i]
        x = preprocess(cv2.imread(path)).transpose(2, 0, 1)
        return (torch.from_numpy(x), torch.tensor(lane, dtype=torch.long),
                torch.tensor([st/STEER_MAX], dtype=torch.float32))   # 회귀


class LaneCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2), nn.ELU(),
            nn.Conv2d(24, 36, 5, 2), nn.ELU(),
            nn.Conv2d(36, 48, 5, 2), nn.ELU(),
            nn.Conv2d(48, 64, 3), nn.ELU(),
            nn.Conv2d(64, 64, 3), nn.ELU(), nn.Flatten())
        self.emb = nn.Embedding(2, 16)          # 차선ID 임베딩
        self.fc = nn.Sequential(
            nn.Dropout(0.3), nn.LazyLinear(100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 10), nn.ELU(),
            nn.Linear(10, 1), nn.Tanh())        # 회귀 [-1,1]
    def forward(self, x, lane):
        f = self.conv(x)
        return self.fc(torch.cat([f, self.emb(lane)], dim=1))


def load_rows():
    rows = []
    # GT CSV(labels_gtL0.csv, labels_gtL1.csv) — 'lane' 컬럼 사용
    for p in sorted(glob.glob(os.path.join(DATA, "labels_gtL*.csv"))):
        for r in csv.DictReader(open(p)):
            ip = os.path.join(IMG_DIR, r["fname"])
            if os.path.exists(ip):
                rows.append((ip, max(-7, min(7, int(r["steering"]))), int(r["lane"])))
    random.Random(SEED).shuffle(rows)
    cnt = collections.Counter(); capped = []
    for ip, st, lane in rows:
        k = (lane, st)
        if cnt[k] < CAP:
            capped.append((ip, st, lane)); cnt[k] += 1
    for lane in (0, 1):
        d = {s: cnt[(lane, s)] for s in range(-7, 8) if cnt[(lane, s)]}
        print(f"lane{lane} 분포:", dict(sorted(d.items())))
    return capped


def main():
    torch.manual_seed(SEED)
    rows = load_rows()
    random.Random(1).shuffle(rows)
    nv = max(1, int(len(rows)*0.1)); val, tr = rows[:nv], rows[nv:]
    print(f"train {len(tr)} / val {len(val)}")
    dev = "cuda:0"
    tl = DataLoader(LaneSet(tr), batch_size=64, shuffle=True, num_workers=4)
    vl = DataLoader(LaneSet(val), batch_size=64, num_workers=2)
    net = LaneCNN().to(dev)
    net(torch.zeros(1, 3, IN_H, IN_W, device=dev), torch.zeros(1, dtype=torch.long, device=dev))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.MSELoss(); best = 1e9
    for ep in range(45):
        net.train(); tre = 0
        for x, lane, y in tl:
            x, lane, y = x.to(dev), lane.to(dev), y.to(dev)
            opt.zero_grad(); l = lossf(net(x, lane), y); l.backward(); opt.step()
            tre += l.item()*len(x)
        net.eval(); mae = {0: 0, 1: 0}; cnt = {0: 0, 1: 0}
        with torch.no_grad():
            for x, lane, y in vl:
                x, lane, y = x.to(dev), lane.to(dev), y.to(dev)
                p = net(x, lane)
                for li in (0, 1):
                    msk = lane == li
                    if msk.any():
                        mae[li] += (p[msk]-y[msk]).abs().sum().item()*STEER_MAX; cnt[li] += msk.sum().item()
        m0 = mae[0]/max(1, cnt[0]); m1 = mae[1]/max(1, cnt[1]); m = (m0+m1)/2
        if m < best:
            best = m
            torch.save({"state_dict": net.state_dict(), "in_w": IN_W, "in_h": IN_H,
                        "steer_max": STEER_MAX}, OUT)
        print(f"ep{ep} train_mse={tre/len(tr):.4f} val_MAE lane1={m0:.2f} lane2={m1:.2f} (best avg {best:.2f})")
    print(f"\n✓ best avg val MAE {best:.2f} → {OUT}")


if __name__ == "__main__":
    main()

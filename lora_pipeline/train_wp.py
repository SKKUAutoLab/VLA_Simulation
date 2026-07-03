#!/usr/bin/env python3
"""
웨이포인트 CNN (차선조건부) — 이미지+lane_id → 미래 ego 웨이포인트 2N개 회귀.
labels_wpL*.csv 사용. 저장: cnn_wp_model.pt
"""
import os, csv, glob, random, math
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from train_cnn import preprocess, IN_W, IN_H

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "dataset")
IMG_DIR = os.path.join(DATA, "images")
OUT = os.path.join(HERE, "cnn_wp_model.pt")
WP_N = 6
WP_SCALE = 5.0   # ego 좌표 정규화(m)
SEED = 0


class WPSet(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        path, wps, lane = self.rows[i]
        x = preprocess(cv2.imread(path)).transpose(2, 0, 1)
        return (torch.from_numpy(x), torch.tensor(lane, dtype=torch.long),
                torch.tensor([v/WP_SCALE for v in wps], dtype=torch.float32))


class WPCNN(nn.Module):
    def __init__(self, nout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2), nn.ELU(),
            nn.Conv2d(24, 36, 5, 2), nn.ELU(),
            nn.Conv2d(36, 48, 5, 2), nn.ELU(),
            nn.Conv2d(48, 64, 3), nn.ELU(),
            nn.Conv2d(64, 64, 3), nn.ELU(), nn.Flatten())
        self.emb = nn.Embedding(2, 16)
        self.fc = nn.Sequential(
            nn.Dropout(0.3), nn.LazyLinear(128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, nout))   # 회귀 (Tanh 없음 — 좌표라 범위 넘을 수 있음)
    def forward(self, x, lane):
        return self.fc(torch.cat([self.conv(x), self.emb(lane)], dim=1))


def load_rows():
    rows = []
    for cf in sorted(glob.glob(os.path.join(DATA, "labels_wpL*.csv"))):
        for r in csv.DictReader(open(cf)):
            ip = os.path.join(IMG_DIR, r["fname"])
            if not os.path.exists(ip):
                continue
            wps = [float(r[f"{c}{k}"]) for k in range(WP_N) for c in ("ex", "ey")]
            rows.append((ip, wps, int(r["lane"])))
    return rows


def main():
    torch.manual_seed(SEED)
    rows = load_rows(); random.Random(1).shuffle(rows)
    nv = max(1, int(len(rows)*0.1)); val, tr = rows[:nv], rows[nv:]
    print(f"train {len(tr)} / val {len(val)} | WP_N={WP_N}")
    dev = "cuda:0"; nout = 2*WP_N
    tl = DataLoader(WPSet(tr), batch_size=64, shuffle=True, num_workers=4)
    vl = DataLoader(WPSet(val), batch_size=64, num_workers=2)
    net = WPCNN(nout).to(dev)
    net(torch.zeros(1, 3, IN_H, IN_W, device=dev), torch.zeros(1, dtype=torch.long, device=dev))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.MSELoss(); best = 1e9
    for ep in range(50):
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
                    m = lane == li
                    if m.any():
                        mae[li] += (p[m]-y[m]).abs().mean(1).sum().item()*WP_SCALE; cnt[li] += m.sum().item()
        m0 = mae[0]/max(1, cnt[0]); m1 = mae[1]/max(1, cnt[1]); mm = (m0+m1)/2
        if mm < best:
            best = mm
            torch.save({"state_dict": net.state_dict(), "in_w": IN_W, "in_h": IN_H,
                        "wp_n": WP_N, "wp_scale": WP_SCALE, "nout": nout}, OUT)
        print(f"ep{ep} train_mse={tre/len(tr):.4f} val_wpMAE(m) lane1={m0:.2f} lane2={m1:.2f} (best {best:.2f})")
    print(f"\n✓ best wpMAE {best:.2f}m → {OUT}")


if __name__ == "__main__":
    main()

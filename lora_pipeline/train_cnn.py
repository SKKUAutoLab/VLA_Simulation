#!/usr/bin/env python3
"""
PilotNet식 소형 CNN — 카메라 이미지 → 조향(회귀). 차선 비전 주행용.
모든 labels*.csv(클린+복구) 사용. 상단(하늘) 크롭 후 200x66 입력.
저장: lora_pipeline/cnn_model.pt  (state_dict + 메타)
"""
import os, csv, glob, math, random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "dataset")
IMG_DIR = os.path.join(DATA, "images")
OUT = os.path.join(HERE, "cnn_model.pt")
IN_W, IN_H = 200, 66
CROP_TOP = 0.45          # 상단 45%(하늘/배경) 버림
STEER_MAX = 7.0
CAP = 350                # 정수 조향값별 상한(편향 완화)
SEED = 0


def preprocess(bgr):
    h = bgr.shape[0]
    bgr = bgr[int(h*CROP_TOP):, :, :]               # 하단만
    bgr = cv2.resize(bgr, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)      # PilotNet=YUV
    return yuv.astype(np.float32) / 255.0           # HWC


class DriveSet(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        path, st = self.rows[i]
        bgr = cv2.imread(path)
        x = preprocess(bgr).transpose(2, 0, 1)       # CHW
        return torch.from_numpy(x), torch.tensor([st/STEER_MAX], dtype=torch.float32)


class PilotNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2), nn.ELU(),
            nn.Conv2d(24, 36, 5, 2), nn.ELU(),
            nn.Conv2d(36, 48, 5, 2), nn.ELU(),
            nn.Conv2d(48, 64, 3), nn.ELU(),
            nn.Conv2d(64, 64, 3), nn.ELU())
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3),
            nn.LazyLinear(100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 10), nn.ELU(),
            nn.Linear(10, 1), nn.Tanh())             # [-1,1]
    def forward(self, x):
        return self.fc(self.conv(x))


def load_rows():
    import collections
    rows = []
    for cf in sorted(glob.glob(os.path.join(DATA, "labels*.csv"))):
        for r in csv.DictReader(open(cf)):
            p = os.path.join(IMG_DIR, r["fname"])
            if os.path.exists(p):
                rows.append((p, max(-7, min(7, int(r["steering"])))))
    # 정수 조향값별 상한
    random.Random(SEED).shuffle(rows)
    cnt = collections.Counter(); capped = []
    for p, st in rows:
        if cnt[st] < CAP:
            capped.append((p, st)); cnt[st] += 1
    print("조향 분포(cap후):", dict(sorted(collections.Counter(s for _, s in capped).items())))
    return capped


def main():
    torch.manual_seed(SEED)
    rows = load_rows()
    random.Random(SEED+1).shuffle(rows)
    nval = max(1, int(len(rows)*0.1))
    val, train = rows[:nval], rows[nval:]
    print(f"train {len(train)} / val {len(val)}")
    dev = "cuda:0"
    tl = DataLoader(DriveSet(train), batch_size=64, shuffle=True, num_workers=4)
    vl = DataLoader(DriveSet(val), batch_size=64, num_workers=2)
    net = PilotNet().to(dev)
    # Lazy 초기화
    net(torch.zeros(1, 3, IN_H, IN_W, device=dev))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.MSELoss()
    best = 1e9
    for ep in range(40):
        net.train(); tr = 0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); out = net(x); l = lossf(out, y); l.backward(); opt.step()
            tr += l.item()*len(x)
        net.eval(); mae = 0; n = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(dev), y.to(dev)
                p = net(x)
                mae += (p-y).abs().sum().item()*STEER_MAX; n += len(x)
        mae /= n
        if mae < best:
            best = mae
            torch.save({"state_dict": net.state_dict(), "in_w": IN_W, "in_h": IN_H,
                        "crop_top": CROP_TOP, "steer_max": STEER_MAX}, OUT)
        print(f"ep{ep} train_mse={tr/len(train):.4f} val_MAE={mae:.2f} (best {best:.2f})")
    print(f"\n✓ best val MAE {best:.2f} (조향 단위) → {OUT}")


if __name__ == "__main__":
    main()

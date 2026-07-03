#!/usr/bin/env python3
"""
수동 시연(manual_demos) 학습 — 사용자 조향을 정답으로 (camera+lane_id → steering).
lane은 pose로 자동 재라벨(둘 다 --lane0으로 찍혔어도 위치로 구분):
  inner중심선 기준 횡오프셋 < THRESH(바깥) → lane0(1차선), 그 외(중심선쪽) → lane1(2차선).
저장: cnn_manual_model.pt (LaneCNN 회귀, train_cnn_lane과 동일 구조)
"""
import os, csv, math, json, random, collections
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from train_cnn import preprocess, IN_W, IN_H, STEER_MAX
from train_cnn_lane import LaneCNN

HERE = os.path.dirname(__file__)
MAN = os.path.join(HERE, "manual_demos")
CSV = os.path.join(MAN, "labels.csv")
IMG_DIR = os.path.join(MAN, "images")
OUT = os.path.join(HERE, "cnn_manual_model.pt")
INNER = os.path.expanduser("~/track_gt_manual.json")
THRESH = -1.4   # 횡오프셋(+안쪽) 이보다 바깥(<)이면 lane0(1차선), 아니면 lane1(2차선)
CAP = 500
SEED = 0


def build_lane_assigner():
    cl = [(float(a), float(b)) for a, b in json.load(open(INNER))["centerline_world"]]
    N = len(cl); cx = sum(p[0] for p in cl)/N; cy = sum(p[1] for p in cl)/N
    def lane_of(px, py):
        i = min(range(N), key=lambda k: (cl[k][0]-px)**2 + (cl[k][1]-py)**2)
        ix, iy = cl[i]; d = math.dist((px, py), (ix, iy))
        inward = (cx-ix)*(px-ix) + (cy-iy)*(py-iy)
        off = d * (1 if inward > 0 else -1)
        return 0 if off < THRESH else 1
    return lane_of


class ManSet(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        path, st, lane = self.rows[i]
        x = preprocess(cv2.imread(path)).transpose(2, 0, 1)
        return (torch.from_numpy(x), torch.tensor(lane, dtype=torch.long),
                torch.tensor([st/STEER_MAX], dtype=torch.float32))


def load_rows():
    lane_of = build_lane_assigner()
    rows = []
    for r in csv.DictReader(open(CSV)):
        ip = os.path.join(IMG_DIR, r["fname"])
        if not os.path.exists(ip):
            continue
        sp = int(r["speed"])
        if sp <= 0:        # 후진/정지 프레임 제외(전진 차선주행만 학습)
            continue
        lane = lane_of(float(r["x"]), float(r["y"]))
        st = max(-7, min(7, int(r["steering"])))
        rows.append((ip, st, lane))
    # (lane,steering) 균형
    random.Random(SEED).shuffle(rows)
    cnt = collections.Counter(); capped = []
    for ip, st, lane in rows:
        if cnt[(lane, st)] < CAP:
            capped.append((ip, st, lane)); cnt[(lane, st)] += 1
    for lane in (0, 1):
        d = {s: cnt[(lane, s)] for s in range(-7, 8) if cnt[(lane, s)]}
        print(f"lane{lane}(1차선=0/2차선=1) n={sum(d.values())} 분포:", dict(sorted(d.items())))
    return capped


def main():
    torch.manual_seed(SEED)
    rows = load_rows(); random.Random(1).shuffle(rows)
    nv = max(1, int(len(rows)*0.1)); val, tr = rows[:nv], rows[nv:]
    print(f"train {len(tr)} / val {len(val)}")
    dev = "cuda:0"
    tl = DataLoader(ManSet(tr), batch_size=64, shuffle=True, num_workers=4)
    vl = DataLoader(ManSet(val), batch_size=64, num_workers=2)
    net = LaneCNN().to(dev)
    net(torch.zeros(1, 3, IN_H, IN_W, device=dev), torch.zeros(1, dtype=torch.long, device=dev))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.MSELoss(); best = 1e9
    for ep in range(50):
        net.train(); tre = 0
        for x, lane, y in tl:
            x, lane, y = x.to(dev), lane.to(dev), y.to(dev)
            opt.zero_grad(); l = lossf(net(x, lane), y); l.backward(); opt.step(); tre += l.item()*len(x)
        net.eval(); mae = {0: 0, 1: 0}; cnt = {0: 0, 1: 0}
        with torch.no_grad():
            for x, lane, y in vl:
                x, lane, y = x.to(dev), lane.to(dev), y.to(dev); p = net(x, lane)
                for li in (0, 1):
                    m = lane == li
                    if m.any(): mae[li] += (p[m]-y[m]).abs().sum().item()*STEER_MAX; cnt[li] += m.sum().item()
        m0 = mae[0]/max(1, cnt[0]); m1 = mae[1]/max(1, cnt[1]); mm = (m0+m1)/2
        if mm < best:
            best = mm
            torch.save({"state_dict": net.state_dict(), "in_w": IN_W, "in_h": IN_H, "steer_max": STEER_MAX}, OUT)
        print(f"ep{ep} mse={tre/len(tr):.4f} val_MAE 1차선={m0:.2f} 2차선={m1:.2f} (best {best:.2f})")
    print(f"\n✓ best {best:.2f} → {OUT}")


if __name__ == "__main__":
    main()

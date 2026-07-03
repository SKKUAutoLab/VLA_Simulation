#!/usr/bin/env python3
"""
수동 시연 WP 학습 — manual_wp_labels.csv(절대경로) → WPCNN(차선조건부).
사용자가 실제 그린 라인의 미래경로를 예측. 저장: cnn_manual_wp_model.pt
드라이브: wp_drive_node.py (MODEL env로 이 파일 지정) + pure-pursuit.
"""
import os, csv, random
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from train_cnn import preprocess, IN_W, IN_H
from train_wp import WPCNN, WP_N, WP_SCALE

HERE = os.path.dirname(__file__)
CSV = os.path.join(HERE, "manual_wp_labels.csv")           # 수동 시연(실제 라인, 절대경로)
RECOV = os.path.join(HERE, "dataset")                       # 복구 텔레포트 라벨(±오프셋)
OUT = os.path.join(HERE, "cnn_manual_wp_model.pt")
SEED = 0


class WPSet(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        path, wps, lane = self.rows[i]
        x = preprocess(cv2.imread(path)).transpose(2, 0, 1)
        return (torch.from_numpy(x), torch.tensor(lane, dtype=torch.long),
                torch.tensor([v/WP_SCALE for v in wps], dtype=torch.float32))


def main():
    torch.manual_seed(SEED)
    rows = []; nman = nrec = 0
    # 1) 수동 시연 WP (절대경로)
    for r in csv.DictReader(open(CSV)):
        if not os.path.exists(r["path"]):
            continue
        wps = [float(r[f"{c}{k}"]) for k in range(WP_N) for c in ("ex", "ey")]
        rows.append((r["path"], wps, int(r["lane"]))); nman += 1
    # 2) 복구 텔레포트 WP (dataset/images/<fname>)
    import glob
    for cf in sorted(glob.glob(os.path.join(RECOV, "labels_wpL*.csv"))):
        for r in csv.DictReader(open(cf)):
            ip = os.path.join(RECOV, "images", r["fname"])
            if not os.path.exists(ip):
                continue
            wps = [float(r[f"{c}{k}"]) for k in range(WP_N) for c in ("ex", "ey")]
            rows.append((ip, wps, int(r["lane"]))); nrec += 1
    print(f"수동 {nman} + 복구 {nrec} = {len(rows)}")
    random.Random(SEED).shuffle(rows)
    nv = max(1, int(len(rows)*0.1)); val, tr = rows[:nv], rows[nv:]
    print(f"train {len(tr)} / val {len(val)}")
    dev = "cuda:0"; nout = 2*WP_N
    tl = DataLoader(WPSet(tr), batch_size=64, shuffle=True, num_workers=4)
    vl = DataLoader(WPSet(val), batch_size=64, num_workers=2)
    net = WPCNN(nout).to(dev)
    net(torch.zeros(1, 3, IN_H, IN_W, device=dev), torch.zeros(1, dtype=torch.long, device=dev))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.MSELoss(); best = 1e9
    for ep in range(60):
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
                    if m.any():
                        mae[li] += (p[m]-y[m]).abs().sum().item()*WP_SCALE; cnt[li] += m.sum().item()*nout
        m0 = mae[0]/max(1, cnt[0]); m1 = mae[1]/max(1, cnt[1]); mm = (m0+m1)/2
        if mm < best:
            best = mm
            torch.save({"state_dict": net.state_dict(), "in_w": IN_W, "in_h": IN_H,
                        "wp_n": WP_N, "wp_scale": WP_SCALE, "nout": nout}, OUT)
        print(f"ep{ep} train_mse={tre/len(tr):.4f} val_wpMAE(m) 1차선={m0:.2f} 2차선={m1:.2f} (best {best:.2f})")
    print(f"\n✓ best wpMAE {best:.2f}m → {OUT}")


if __name__ == "__main__":
    main()

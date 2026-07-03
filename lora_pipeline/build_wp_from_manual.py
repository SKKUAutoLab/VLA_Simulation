#!/usr/bin/env python3
"""
수동 시연 WP 라벨 — 사용자가 실제 본 카메라 이미지 + 그가 따른 차선의 미래경로.
각 프레임의 실제 pose(px,py,yaw)에서 해당 차선 중심선의 미래 N점을 ego 좌표로.
(차가 라인에서 약간 벗어난 프레임은 자연히 '복귀' WP가 라벨됨 → closed-loop 강건)
lane은 pose로 판정: 1차선=바깥(track_gt_outward), 2차선=중심선(track_gt_manual).
출력: manual_wp_labels.csv  (path, ex0..ey5, lane)  — train_manual_wp.py 가 읽음.
"""
import os, csv, math, json

HERE = os.path.dirname(__file__)
MAN = os.path.join(HERE, "manual_demos")
CSV = os.path.join(MAN, "labels.csv")
IMG_DIR = os.path.join(MAN, "images")
OUT = os.path.join(HERE, "manual_wp_labels.csv")
INNER = os.path.expanduser("~/track_gt_manual.json")          # 2차선(lane1)
OUTWARD = os.path.expanduser("~/track_gt_outward_centerline.json")  # 1차선(lane0)
YAW_OFFSET = math.pi / 2
WP_N = 6
WP_STRIDE = 5      # 중심선 인덱스 간격(~0.77m)
THRESH = -1.4      # inner기준 부호오프셋<이값 → lane0(1차선,바깥)


def load_cl(p):
    return [(float(a), float(b)) for a, b in json.load(open(p))["centerline_world"]]


def nearest(cl, x, y):
    return min(range(len(cl)), key=lambda k: (cl[k][0]-x)**2 + (cl[k][1]-y)**2)


def ego_wp(px, py, yaw, cl, i0, step):
    fwd = yaw - YAW_OFFSET; cf, sf = math.cos(fwd), math.sin(fwd); n = len(cl); out = []
    for k in range(1, WP_N+1):
        wx, wy = cl[(i0 + k*step) % n]
        dx, dy = wx-px, wy-py
        out += [cf*dx + sf*dy, -sf*dx + cf*dy]   # 전방ex, 좌ey
    return out


def main():
    inner = load_cl(INNER); outward = load_cl(OUTWARD)
    cx = sum(p[0] for p in inner)/len(inner); cy = sum(p[1] for p in inner)/len(inner)
    # 학습 규약: lane0=1차선(중심선=inner), lane1=2차선(바깥=outward)
    CLS = {0: inner, 1: outward}

    def lane_of(px, py):
        i = nearest(inner, px, py); ix, iy = inner[i]; d = math.dist((px, py), (ix, iy))
        inward = (cx-ix)*(px-ix) + (cy-iy)*(py-iy)
        off = d if inward > 0 else -d
        return 1 if off < THRESH else 0   # 바깥(off<-1.4)=2차선(1), 그 외 중심=1차선(0)

    hdr = ["path"] + [f"{c}{k}" for k in range(WP_N) for c in ("ex", "ey")] + ["lane"]
    f = open(OUT, "w", newline=""); w = csv.writer(f); w.writerow(hdr)
    cnt = {0: 0, 1: 0}; bad_dir = 0
    for r in csv.DictReader(open(CSV)):
        if int(r["speed"]) <= 0:
            continue
        ip = os.path.join(IMG_DIR, r["fname"])
        if not os.path.exists(ip):
            continue
        px, py, yaw = float(r["x"]), float(r["y"]), float(r["yaw"])
        lane = lane_of(px, py); cl = CLS[lane]
        i0 = nearest(cl, px, py)
        ego = ego_wp(px, py, yaw, cl, i0, WP_STRIDE)
        if ego[0] < 0:        # 전방 WP가 후방 → 차가 인덱스 역방향 주행 → 역walk로 재시도
            ego = ego_wp(px, py, yaw, cl, i0, -WP_STRIDE)
            if ego[0] < 0:    # 그래도 후방이면(코너 등) 스킵
                bad_dir += 1; continue
        w.writerow([ip] + [f"{v:.3f}" for v in ego] + [lane]); cnt[lane] += 1
    f.close()
    print(f"✓ {OUT}: 1차선(lane0,중심)={cnt[0]} 2차선(lane1,바깥)={cnt[1]} | 스킵 {bad_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
사용자 수동 시연에서 차선 중심선을 도출(합성 오프셋 X — 실제 주행 라인).
inner 중심선 인덱스별로 해당 차선 시연점을 평균 → on-road 실제 라인.
빈 인덱스는 보간, 살짝 스무딩. lane0=1차선(바깥), lane1=2차선(중심선쪽).
출력: ~/track_gt_lane0_demo.json, ~/track_gt_lane1_demo.json
"""
import os, csv, math, json
import numpy as np

HERE = os.path.dirname(__file__)
CSV = os.path.join(HERE, "manual_demos", "labels.csv")
INNER = os.path.expanduser("~/track_gt_manual.json")
THRESH = -1.4
OUT = {0: os.path.expanduser("~/track_gt_lane0_demo.json"),
       1: os.path.expanduser("~/track_gt_lane1_demo.json")}


def main():
    inner = [(float(a), float(b)) for a, b in json.load(open(INNER))["centerline_world"]]
    N = len(inner)
    cx = sum(p[0] for p in inner)/N; cy = sum(p[1] for p in inner)/N

    def nidx(x, y): return min(range(N), key=lambda k: (inner[k][0]-x)**2 + (inner[k][1]-y)**2)

    def lane_of(x, y):
        i = nidx(x, y); ix, iy = inner[i]; d = math.dist((x, y), (ix, iy))
        inward = (cx-ix)*(x-ix) + (cy-iy)*(y-iy)
        return 0 if (d if inward > 0 else -d) < THRESH else 1

    bins = {0: [[] for _ in range(N)], 1: [[] for _ in range(N)]}
    for r in csv.DictReader(open(CSV)):
        if int(r["speed"]) <= 0:
            continue
        x, y = float(r["x"]), float(r["y"]); lane = lane_of(x, y)
        bins[lane][nidx(x, y)].append((x, y))

    for lane in (0, 1):
        pts = []
        for i in range(N):
            b = bins[lane][i]
            pts.append((sum(p[0] for p in b)/len(b), sum(p[1] for p in b)/len(b)) if b else None)
        # 빈 인덱스 원형 보간
        filled = list(pts)
        for i in range(N):
            if filled[i] is None:
                lo = next(j for j in range(1, N) if pts[(i-j) % N]); hi = next(j for j in range(1, N) if pts[(i+j) % N])
                a = pts[(i-lo) % N]; b = pts[(i+hi) % N]; t = lo/(lo+hi)
                filled[i] = (a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1]))
        # 원형 가우시안 스무딩(±W) — 시연 산포 제거, 매끄러운 기준/타깃 라인
        W = 8
        wts = np.array([math.exp(-(k*k)/(2*(W/2.0)**2)) for k in range(-W, W+1)])
        wts /= wts.sum()
        arr = np.array(filled); sm = arr.copy()
        for i in range(N):
            sm[i] = sum(wts[j]*arr[(i+k) % N] for j, k in enumerate(range(-W, W+1)))
        line = sm.tolist()
        xs = [p[0] for p in line]; ys = [p[1] for p in line]
        miss = sum(1 for p in pts if p is None)
        print(f"lane{lane}('{'1차선바깥' if lane==0 else '2차선중심'}'): {N}pts 빈칸{miss} "
              f"x[{min(xs):.1f},{max(xs):.1f}] y[{min(ys):.1f},{max(ys):.1f}]")
        json.dump({"meta": {"src": "demo", "lane": lane}, "centerline_world": line}, open(OUT[lane], "w"))
    print("✓ 저장:", OUT[0], OUT[1])


if __name__ == "__main__":
    main()

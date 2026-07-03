#!/usr/bin/env python3
"""
바깥 차선 중심선 = TRACK_CENTERLINE 에서 루프중심 반대(바깥)쪽으로 SEP 오프셋.
사용자 수동 시연의 '1차선'(실측 평균 -2.8m, 바깥)에 해당.
저장: ~/track_gt_outward_centerline.json
"""
import json, math, os

GT = os.path.expanduser("~/track_gt_manual.json")
OUT = os.path.expanduser("~/track_gt_outward_centerline.json")
SEP = 2.8


def main():
    cl = [(float(a), float(b)) for a, b in json.load(open(GT))["centerline_world"]]
    n = len(cl)
    cx = sum(p[0] for p in cl)/n; cy = sum(p[1] for p in cl)/n
    out = []
    for i in range(n):
        x0, y0 = cl[i]; nx, ny = cl[(i+1) % n]
        tang = math.atan2(ny-y0, nx-x0)
        perp = (math.cos(tang+math.pi/2), math.sin(tang+math.pi/2))
        to_c = (cx-x0, cy-y0)
        if perp[0]*to_c[0]+perp[1]*to_c[1] > 0:   # 중심을 향하면 반대로(=바깥)
            perp = (-perp[0], -perp[1])
        out.append([x0+SEP*perp[0], y0+SEP*perp[1]])
    xs = [p[0] for p in out]; ys = [p[1] for p in out]
    print(f"outward {n}pts | x[{min(xs):.1f},{max(xs):.1f}] y[{min(ys):.1f},{max(ys):.1f}] (바깥 {SEP}m)")
    json.dump({"meta": {"lane": "outward", "sep": SEP}, "centerline_world": out}, open(OUT, "w"))
    print("✓ 저장 →", OUT)


if __name__ == "__main__":
    main()

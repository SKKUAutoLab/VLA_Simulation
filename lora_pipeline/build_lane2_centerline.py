#!/usr/bin/env python3
"""
2차선(lane2) 중앙선 = 1차선(TRACK_CENTERLINE) 에서 LANE_SEPARATION 만큼
infield(루프 중심) 쪽으로 수직 오프셋. (track_data.py 기준)
저장: ~/track_gt_lane2_centerline.json
"""
import json, math, os

GT = os.path.expanduser("~/track_gt_manual.json")
OUT = os.path.expanduser("~/track_gt_lane2_centerline.json")
SEP = 2.5   # track_data.LANE_SEPARATION


def main():
    cl = [(float(a), float(b)) for a, b in json.load(open(GT))["centerline_world"]]
    n = len(cl)
    cx = sum(p[0] for p in cl)/n; cy = sum(p[1] for p in cl)/n   # 루프 중심
    lane2 = []
    for i in range(n):
        x0, y0 = cl[i]; nx, ny = cl[(i+1) % n]
        tang = math.atan2(ny-y0, nx-x0)
        # 수직 단위벡터 2개 중 중심을 향하는 쪽 선택
        perp = (math.cos(tang+math.pi/2), math.sin(tang+math.pi/2))
        to_c = (cx-x0, cy-y0)
        if perp[0]*to_c[0]+perp[1]*to_c[1] < 0:
            perp = (-perp[0], -perp[1])
        lane2.append([x0+SEP*perp[0], y0+SEP*perp[1]])
    xs = [p[0] for p in lane2]; ys = [p[1] for p in lane2]
    print(f"lane2 {n}pts | x[{min(xs):.1f},{max(xs):.1f}] y[{min(ys):.1f},{max(ys):.1f}] (infield쪽 {SEP}m)")
    json.dump({"meta": {"lane": "lane2", "sep": SEP}, "centerline_world": lane2}, open(OUT, "w"))
    print("✓ 저장 →", OUT)


if __name__ == "__main__":
    main()

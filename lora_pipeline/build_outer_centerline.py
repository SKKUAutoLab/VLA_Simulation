#!/usr/bin/env python3
"""
2차선(outer) 중심선 생성 — track_gt_edit_state.json 의 outer_inn/outer_out 경계
(픽셀 폴리곤)을 world로 변환, inner centerline 순서에 맞춰 중간선을 만든다.
저장: ~/track_gt_outer_centerline.json {"centerline_world":[[x,y],...]}
"""
import json, math, os

GT_MANUAL = os.path.expanduser("~/track_gt_manual.json")
GT_EDIT   = os.path.expanduser("~/track_gt_edit_state.json")
OUT       = os.path.expanduser("~/track_gt_outer_centerline.json")


def to_world(px, py):
    # meta.formula (축 스왑 주의): world_x=f(py), world_y=f(px)
    return (-20.237 + (py/884)*40.473, -26.915 + (px/1180)*53.83)


def nearest(pt, pts):
    return min(pts, key=lambda q: (q[0]-pt[0])**2 + (q[1]-pt[1])**2)


def main():
    inner = [(float(x), float(y)) for x, y in
             json.load(open(GT_MANUAL))["centerline_world"]]
    ep = json.load(open(GT_EDIT))["edit_polygons"]
    oin = [to_world(p[0], p[1]) for p in ep["outer_inn"]]
    oout = [to_world(p[0], p[1]) for p in ep["outer_out"]]

    # inner 각 점에 대해 outer 두 경계의 최근접점 중점 = outer 차선 중심
    outer = []
    for p in inner:
        a = nearest(p, oin)
        b = nearest(p, oout)
        outer.append([(a[0]+b[0])/2, (a[1]+b[1])/2])

    # 간격/오프셋 점검
    d = [math.dist(inner[i], outer[i]) for i in range(len(inner))]
    print(f"outer centerline {len(outer)}pts | inner↔outer 간격 평균 "
          f"{sum(d)/len(d):.2f}m (min {min(d):.2f} max {max(d):.2f})")
    xs = [q[0] for q in outer]; ys = [q[1] for q in outer]
    print(f"outer x[{min(xs):.1f},{max(xs):.1f}] y[{min(ys):.1f},{max(ys):.1f}]")

    json.dump({"meta": {"lane": "outer", "derived_from": "edit_polygons"},
               "centerline_world": outer}, open(OUT, "w"))
    print(f"✓ 저장 → {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
사용자 주석(~/track_gt_edit_state.json)의 차선 경계 폴리곤에서 각 차선 중앙선 추출.
inner 차선 = midline(inner_inn, inner_out) = 1차선(=TRACK_CENTERLINE)
outer 차선 = midline(outer_inn, outer_out) = 2차선
픽셀→월드: gt_annotator.pixel_to_world. 720점 균일 리샘플 + 가우시안 스무딩.
저장: lane_id0=~/track_gt_lane1_center.json(1차선), lane_id1=~/track_gt_lane2_center.json(2차선)
"""
import json, os, math
import numpy as np

IMG_W, IMG_H = 1180.0, 884.0
def p2w(px, py): return (-20.237 + (py/IMG_H)*40.473, -26.915 + (px/IMG_W)*53.83)
EDIT = os.path.expanduser("~/track_gt_edit_state.json")
OUT = {0: os.path.expanduser("~/track_gt_lane1_center.json"),   # 1차선=inner
       1: os.path.expanduser("~/track_gt_lane2_center.json")}   # 2차선=outer
NP = 720


def to_world(poly): return [p2w(px, py) for px, py in poly]


def midline(bndA, bndB):
    """경계 A 각 점에 대해 B의 최근접 점과의 중점 → 차선 중앙선(순서 A 따름)."""
    B = np.array(bndB)
    out = []
    for ax, ay in bndA:
        j = int(np.argmin((B[:, 0]-ax)**2 + (B[:, 1]-ay)**2))
        out.append(((ax+B[j][0])/2, (ay+B[j][1])/2))
    return out


def resample_closed(pts, n):
    p = np.array(pts + [pts[0]])           # 닫기
    seg = np.sqrt(((p[1:]-p[:-1])**2).sum(1)); cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]; targets = np.linspace(0, total, n, endpoint=False)
    res = []
    for t in targets:
        k = np.searchsorted(cum, t) - 1; k = max(0, min(k, len(seg)-1))
        u = (t-cum[k])/seg[k] if seg[k] > 1e-9 else 0
        res.append(p[k] + u*(p[k+1]-p[k]))
    return np.array(res)


def smooth_closed(arr, W=6):
    wts = np.array([math.exp(-(k*k)/(2*(W/2.0)**2)) for k in range(-W, W+1)]); wts /= wts.sum()
    n = len(arr); sm = arr.copy()
    for i in range(n):
        sm[i] = sum(wts[j]*arr[(i+k) % n] for j, k in enumerate(range(-W, W+1)))
    return sm


def aligned_to_ref(cl, ref):
    """cl의 진행방향을 ref(주행방향)와 맞춤. 역방향이면 뒤집기."""
    def fwd(c, i): n = len(c); return c[(i+5) % n]-c[i]
    def nidx(c, p): return int(np.argmin(((c-p)**2).sum(1)))
    refa = np.array(ref); dots = []
    for i in range(0, len(cl), 30):
        j = nidx(refa, cl[i]); fa = fwd(cl, i); fr = refa[(j+5) % len(refa)]-refa[j]
        na = np.hypot(*fa); nr = np.hypot(*fr)
        if na > 0 and nr > 0: dots.append((fa@fr)/(na*nr))
    if np.mean(dots) < 0:
        return cl[::-1].copy()
    return cl


def main():
    ep = json.load(open(EDIT))["edit_polygons"]
    ref = [(float(a), float(b)) for a, b in json.load(open(os.path.expanduser("~/track_gt_manual.json")))["centerline_world"]]
    lanes = {0: ("inner_inn", "inner_out"), 1: ("outer_inn", "outer_out")}
    for lid, (a, b) in lanes.items():
        cl = midline(to_world(ep[a]), to_world(ep[b]))
        cl = smooth_closed(resample_closed(cl, NP))
        cl = aligned_to_ref(cl, ref)       # 주행방향(TRACK_CENTERLINE) 정렬
        xs = cl[:, 0]; ys = cl[:, 1]
        nm = "1차선(inner)" if lid == 0 else "2차선(outer)"
        print(f"{nm}: {NP}pts x[{xs.min():.1f},{xs.max():.1f}] y[{ys.min():.1f},{ys.max():.1f}]")
        json.dump({"meta": {"src": "annotation", "lane": lid},
                   "centerline_world": cl.tolist()}, open(OUT[lid], "w"))
    # 1차선이 TRACK_CENTERLINE과 일치하는지
    inner = [(float(a), float(b)) for a, b in json.load(open(os.path.expanduser("~/track_gt_manual.json")))["centerline_world"]]
    l1 = json.load(open(OUT[0]))["centerline_world"]
    e = [min(math.dist(p, q) for q in inner) for p in l1]
    print(f"검증: 1차선중앙 vs TRACK_CENTERLINE 평균 {sum(e)/len(e):.2f}m max {max(e):.2f}m (작아야 일치)")
    print("✓ 저장:", OUT[0], OUT[1])


if __name__ == "__main__":
    main()

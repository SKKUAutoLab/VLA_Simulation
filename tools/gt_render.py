#!/usr/bin/env python3
"""
gt_render.py — GT 차선 중심선을 월드좌표(X/Y, m) 평면에 플롯.

사용법:
    python3 tools/gt_render.py [GT_JSON] [-o OUT_PNG]

예:
    python3 tools/gt_render.py ~/track_gt_manual.json -o /tmp/gt_render.png
    python3 tools/gt_render.py                       # 기본 ~/track_gt_manual.json
"""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEF_IN = os.path.expanduser("~/track_gt_manual.json")


def main():
    ap = argparse.ArgumentParser(description="Plot GT centerline in world coords")
    ap.add_argument("gt_json", nargs="?", default=DEF_IN, help="GT json (centerline_world)")
    ap.add_argument("-o", "--out", default="/tmp/gt_render.png", help="output png path")
    a = ap.parse_args()

    d = json.load(open(os.path.expanduser(a.gt_json)))
    pts = d["centerline_world"]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    plt.figure(figsize=(10, 10))
    plt.plot(xs + [xs[0]], ys + [ys[0]], "-", color="gold", lw=1.5, label="GT centerline")
    plt.scatter(xs, ys, s=6, c="red", zorder=3, label="GT points")
    plt.scatter([xs[0]], [ys[0]], s=120, c="lime", marker="*", zorder=4, label="start")
    plt.gca().set_aspect("equal")
    plt.grid(alpha=0.3)
    plt.title(f"{os.path.basename(a.gt_json)}  ({len(pts)} pts)")
    plt.xlabel("world X (m)")
    plt.ylabel("world Y (m)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(a.out, dpi=130)
    print("saved", a.out)


if __name__ == "__main__":
    main()

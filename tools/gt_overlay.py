#!/usr/bin/env python3
"""
gt_overlay.py — GT 차선 중심선을 실제 트랙 이미지(track.png) 위에 오버레이 (OpenCV).

GUI의 world_to_pixel() 변환식과 동일하게 월드좌표를 픽셀로 변환해 그린다.
GT json에 centerline_pixels가 있으면 그대로 사용, 없으면 centerline_world를 변환.

사용법:
    python3 tools/gt_overlay.py [GT_JSON] [-o OUT_PNG]

예:
    python3 tools/gt_overlay.py ~/track_gt_manual.json -o /tmp/gt_overlay.png
"""
import argparse, json, os
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRACK = os.path.join(
    HERE, "..", "src", "simulation_pkg", "models", "race_track",
    "materials", "textures", "track.png")
DEF_IN = os.path.expanduser("~/track_gt_manual.json")


def main():
    ap = argparse.ArgumentParser(description="Overlay GT centerline on track.png")
    ap.add_argument("gt_json", nargs="?", default=DEF_IN)
    ap.add_argument("-o", "--out", default="/tmp/gt_overlay.png")
    ap.add_argument("--track", default=TRACK, help="track background image")
    a = ap.parse_args()

    img = cv2.imread(a.track)
    if img is None:
        raise SystemExit(f"track image not found: {a.track}")
    H, W = img.shape[:2]

    def w2p(wx, wy):
        return int((wy + 26.915) / 53.83 * W), int((wx + 20.237) / 40.473 * H)

    d = json.load(open(os.path.expanduser(a.gt_json)))
    if d.get("centerline_pixels"):
        px = [tuple(p) for p in d["centerline_pixels"]]
    else:
        px = [w2p(wx, wy) for wx, wy in d["centerline_world"]]

    cv2.polylines(img, [np.array(px, np.int32)], True, (0, 255, 255), 2, cv2.LINE_AA)
    for x, y in px:
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img, tuple(px[0]), 10, (0, 255, 0), 2, cv2.LINE_AA)  # start

    label = f"GT: {os.path.basename(a.gt_json)} ({len(px)} pts)"
    cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(a.out, img)
    print("saved", a.out, f"({W}x{H})")


if __name__ == "__main__":
    main()

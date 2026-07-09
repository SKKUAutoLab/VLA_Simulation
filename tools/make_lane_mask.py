#!/usr/bin/env python3
"""트랙 위에 1차선/2차선 주행 영역을 색 밴드로 칠한 '차선 마스킹' 이미지 생성.
어노테이터 GUI 스크린샷과 달리, 실제 차선 영역을 명확히 구분한 마스크를 만든다.

사용:  python3 tools/make_lane_mask.py [-o OUT] [--band 52]
입력:  ~/track_gt_lane1_demo.json(1차선), ~/track_gt_lane0_demo.json(2차선),
       src/simulation_pkg/models/race_track/materials/textures/track.png
"""
import argparse, json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

IMG_W, IMG_H = 1180.0, 884.0
HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
TRACK = os.path.join(WS, "src/simulation_pkg/models/race_track/materials/textures/track.png")
LANES = [("1차선 (inner)", "~/track_gt_lane1_demo.json", (230, 30, 30)),   # 빨강
         ("2차선 (outer)", "~/track_gt_lane0_demo.json", (30, 90, 230))]   # 파랑


def w2p(wx, wy):
    return (int((wy + 26.915) / 53.83 * IMG_W), int((wx + 20.237) / 40.473 * IMG_H))


def smooth_closed(P, win):
    """폐곡선 주기적 이동평균 — GT 중심선의 직선구간 지터 제거(시각화용)."""
    if win < 3:
        return P
    k = np.ones(win) / win
    Pp = np.vstack([P[-win:], P, P[:win]])
    xs = np.convolve(Pp[:, 0], k, mode="same"); ys = np.convolve(Pp[:, 1], k, mode="same")
    return np.stack([xs, ys], 1)[win:-win]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=os.path.expanduser("~/Downloads/track_lane_mask.png"))
    ap.add_argument("--band", type=int, default=52, help="차선 밴드 두께(px) ≈ 2.4m")
    ap.add_argument("--smooth", type=int, default=21, help="중심선 스무딩 창(0=끄기). GT 직선구간 지터 제거")
    a = ap.parse_args()

    bg = Image.open(TRACK).convert("RGBA")
    if bg.size != (int(IMG_W), int(IMG_H)):
        bg = bg.resize((int(IMG_W), int(IMG_H)))
    overlay = Image.new("RGBA", bg.size, (0, 0, 0, 0)); od = ImageDraw.Draw(overlay)
    for _, path, col in LANES:
        d = json.load(open(os.path.expanduser(path)))
        P = np.array(d["centerline_world"])
        P = smooth_closed(P, a.smooth)
        pts = [w2p(wx, wy) for wx, wy in P]; pts.append(pts[0])
        od.line(pts, fill=col + (120,), width=a.band, joint="curve")  # 반투명 밴드(마스크)
        od.line(pts, fill=col + (255,), width=4, joint="curve")       # 중심선
    out = Image.alpha_composite(bg, overlay).convert("RGB")

    dr = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 30)
    except Exception:
        font = ImageFont.load_default()
    dr.rectangle([20, 20, 360, 110], fill=(255, 255, 255))
    for k, (name, _, col) in enumerate(LANES):
        y = 35 + k * 37
        dr.rectangle([35, y, 75, y + 25], fill=col); dr.text((85, y - 2), name, fill=(20, 20, 20), font=font)
    out.save(a.out); print("saved", a.out, out.size)


if __name__ == "__main__":
    main()

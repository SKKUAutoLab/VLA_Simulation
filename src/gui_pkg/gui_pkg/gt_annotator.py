#!/usr/bin/env python3
"""
gt_annotator.py  v4
===================
track.png → 도로 검출 → 차선 분리 → GT 저장  3단계 워크플로우

[픽셀 ↔ 세계 좌표]
  world_x = -20.237 + (py / 884) * 40.473
  world_y = -26.915 + (px / 1180) * 53.83

[알려진 픽셀 색상]
  도로:     BGR≈(121,118,116)  HSV V≈121
  배경:     BGR≈( 72, 71, 76)  HSV V≈76
  흰 차선:  BGR≈(231,231,230)  HSV V≈231
"""

import sys, json, os, math
import numpy as np
import cv2

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QGroupBox, QScrollArea,
    QFileDialog, QMessageBox, QCheckBox, QSpinBox,
    QStatusBar, QToolBar, QButtonGroup, QRadioButton,
    QFrame, QComboBox, QSizePolicy,
)
from PySide6.QtCore import Qt, QPoint, Signal, QThread
from PySide6.QtGui import (
    QPixmap, QPainter, QPen, QBrush, QColor, QImage,
    QAction, QFont, QKeySequence, QPolygon,
)
from PySide6.QtCore import QPoint as _QPoint  # alias (QPoint already imported)

# ─── 좌표 변환 ────────────────────────────────────────────────────────────────
IMG_W = 1180.0
IMG_H = 884.0

def pixel_to_world(px: float, py: float):
    return -20.237 + (py / IMG_H) * 40.473, -26.915 + (px / IMG_W) * 53.83

def world_to_pixel(wx: float, wy: float):
    return int((wy + 26.915) / 53.83 * IMG_W), int((wx + 20.237) / 40.473 * IMG_H)

LANDMARKS = {
    "Start":      (-2.55, -22.71),
    "Traffic_LT": (-5.63,  17.90),
    "Obstacle_1": (-3.66,   8.71),
    "Obstacle_2": (-3.66,   2.04),
}

# ─── 정적 트랙 피처 (주차칸 / IN·OUT선 / 횡단보도 / 출발점) ────────────────────
# 차량(prius) 앞우측 휠 중심 — 모델 로컬 (전방 = 모델 -y, 좌측 = +x)
WHEEL_FR_LOCAL = (-0.760002, -1.41)
START_YAW = 0.0   # IN 출발 heading: '← IN' 방향(서쪽)

# 피처 종류별 표시 색 (R, G, B)
FEAT_COLORS = {
    "vertical_parking": (80, 230, 120),
    "parallel_parking": (60, 200, 230),
    "crosswalk":        (240, 220, 60),
    "in_line":          (255, 150, 40),
    "out_line":         (230, 90, 230),
    "start_point":      (255, 70, 70),
    "obstacle":         (255, 130, 0),
    "traffic_light":    (210, 70, 255),
}

DEFAULT_FEATURES_PATH = os.path.expanduser("~/track_features.json")

# ── 정적 객체 좌표 (src/simulation_pkg/.../012_deploy_lib.py 에서 가져옴, Gazebo 월드) ──
TRAFFIC_LIGHTS = [
    {"id": "TL1", "world": (-5.6255, 17.9036), "yaw": 1.568773,
     "src": "traffic_light_stand"},
]
OBSTACLES = [
    {"id": "OB_fix1", "world": (-3.659642, 8.710748),  "yaw": -0.013934, "src": "obstacle_coordinates_1"},
    {"id": "OB_fix2", "world": (-3.659642, 2.037476),  "yaw": -0.013934, "src": "obstacle_coordinates_2"},
    {"id": "OB_av1",  "world": (12.251981, -15.909271), "yaw": 2.484252, "src": "obstacle_coordinates1"},
    {"id": "OB_p2a",  "world": (11.884767, 11.605120), "yaw": 3.25, "src": "obstacle_coordinates2"},
    {"id": "OB_p2b",  "world": (12.040719, 10.060495), "yaw": 3.25, "src": "obstacle_coordinates2"},
    {"id": "OB_p2c",  "world": (12.230394, 8.181866),  "yaw": 3.25, "src": "obstacle_coordinates2"},
    {"id": "OB_p3a",  "world": (16.106836, -0.111269), "yaw": 3.25, "src": "obstacle_coordinates3"},
    {"id": "OB_p3b",  "world": (16.281788, -1.844067), "yaw": 3.25, "src": "obstacle_coordinates3"},
    {"id": "OB_p3c",  "world": (16.366463, -3.446680), "yaw": 3.25, "src": "obstacle_coordinates3"},
]
# OUT측 출발점 — 012_deploy_lib.parking_start (base_link 스폰 pose, yaw≈-π=동쪽)
OUT_START_SPAWNS = [
    (-1.672862, -16.311572, -3.133789),
    (-1.681217, -15.244810, -3.133795),
    (-0.772971, -15.237810, -3.133797),
    (-0.764668, -16.302988, -3.133797),
]


def spawn_pose_from_dot(px, py, yaw=START_YAW):
    """오른쪽 앞바퀴 중앙이 (px,py)에 오는 base_link 스폰 pose (x,y,yaw).
       spawn = world_dot − Rz(yaw)·wheel_local."""
    wx, wy = pixel_to_world(px, py)
    lx, ly = WHEEL_FR_LOCAL
    c, s = math.cos(yaw), math.sin(yaw)
    return wx - (c * lx - s * ly), wy - (s * lx + c * ly), yaw


def dot_from_spawn(sx, sy, yaw):
    """base_link 스폰 pose → 오른쪽 앞바퀴 중앙 픽셀점 (역변환)."""
    lx, ly = WHEEL_FR_LOCAL
    c, s = math.cos(yaw), math.sin(yaw)
    return world_to_pixel(sx + (c * lx - s * ly), sy + (s * lx + c * ly))


def parking_target_poses(kind: str, cx: float, cy: float) -> dict:
    """주차칸 중심(cx,cy world) → 전면/후면 주차 목표 base_link pose.
       수직주차: 베이 깊이축=world_x (전면=북쪽 -x, 후면=남쪽 +x)
       평행주차: 슬롯 길이축=world_y (front/rear = 두 진행방향)"""
    if kind == "vertical_parking":
        front_yaw, rear_yaw = -math.pi / 2, math.pi / 2
    else:  # parallel_parking
        front_yaw, rear_yaw = 0.0, math.pi
    return {
        "front": {"desc": "전면주차(nose-in)",
                  "base_pose": {"x": round(cx, 4), "y": round(cy, 4),
                                "yaw": round(front_yaw, 6)}},
        "rear":  {"desc": "후면주차(back-in)",
                  "base_pose": {"x": round(cx, 4), "y": round(cy, 4),
                                "yaw": round(rear_yaw, 6)}},
    }


def build_default_features() -> list:
    """track.png 측정값 기반 기본 피처 (픽셀 기하). 자동 검출 로드용."""
    feats = []
    V_DIV, V_T, V_B = [521, 596, 670, 743, 817], 311, 425
    for i in range(4):
        x0, x1 = V_DIV[i], V_DIV[i + 1]
        feats.append({"kind": "vertical_parking", "type": "box", "id": f"V{i+1}",
                      "pts": [(x0, V_T), (x1, V_T), (x1, V_B), (x0, V_B)]})
    P_DIV, P_T, P_B = [364, 478, 592, 707, 822], 565, 632
    for i in range(4):
        x0, x1 = P_DIV[i], P_DIV[i + 1]
        feats.append({"kind": "parallel_parking", "type": "box", "id": f"P{i+1}",
                      "pts": [(x0, P_T), (x1, P_T), (x1, P_B), (x0, P_B)]})
    feats.append({"kind": "out_line", "type": "line", "id": "OUT",
                  "pts": [(286, 440), (286, 548)]})
    feats.append({"kind": "in_line", "type": "line", "id": "IN",
                  "pts": [(895, 440), (895, 549)]})
    feats.append({"kind": "crosswalk", "type": "box", "id": "CW",
                  "pts": [(1010, 340), (1120, 340), (1120, 409), (1010, 409)]})
    # IN측 출발점 (텍스처 점, yaw=0=서쪽)
    for i, (x, y) in enumerate([(914, 483), (938, 483), (914, 508), (938, 508)]):
        feats.append({"kind": "start_point", "type": "point", "id": f"IN{i+1}",
                      "side": "IN", "pts": [(x, y)], "yaw": START_YAW})
    # OUT측 출발점 (parking_start 스폰 pose 역산, yaw≈-π=동쪽). spawn=원본 정확값
    for i, (sx, sy, syaw) in enumerate(OUT_START_SPAWNS):
        feats.append({"kind": "start_point", "type": "point", "id": f"OUT{i+1}",
                      "side": "OUT", "pts": [dot_from_spawn(sx, sy, syaw)],
                      "yaw": syaw, "spawn": (sx, sy, syaw)})
    # 장애물 / 신호등 (월드좌표 → 픽셀). world=원본 정확값
    for ob in OBSTACLES:
        feats.append({"kind": "obstacle", "type": "point", "id": ob["id"],
                      "pts": [world_to_pixel(*ob["world"])],
                      "yaw": ob["yaw"], "src": ob["src"], "world": ob["world"]})
    for tl in TRAFFIC_LIGHTS:
        feats.append({"kind": "traffic_light", "type": "point", "id": tl["id"],
                      "pts": [world_to_pixel(*tl["world"])],
                      "yaw": tl["yaw"], "src": tl["src"], "world": tl["world"]})
    return feats


def feature_to_world(f: dict) -> dict:
    """편집 피처(픽셀) → 월드 좌표 직렬화 dict."""
    pts = f["pts"]
    out = {"id": f["id"], "kind": f["kind"], "type": f["type"]}
    if f["type"] == "box":
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        wx, wy = pixel_to_world(cx, cy)
        out["center_world"] = [round(wx, 4), round(wy, 4)]
        out["corners_world"] = [[round(a, 4), round(b, 4)]
                                for a, b in (pixel_to_world(x, y) for x, y in pts)]
        out["bbox_pixel"] = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
        out["size_world"] = {
            "x": round(abs(max(ys) - min(ys)) / IMG_H * 40.473, 4),
            "y": round(abs(max(xs) - min(xs)) / IMG_W * 53.83, 4)}
        if f["kind"] in ("vertical_parking", "parallel_parking"):
            out["target_poses"] = parking_target_poses(f["kind"], wx, wy)
    elif f["type"] == "line":
        out["endpoints_pixel"] = [[int(x), int(y)] for x, y in pts]
        out["endpoints_world"] = [[round(a, 4), round(b, 4)]
                                  for a, b in (pixel_to_world(x, y) for x, y in pts)]
        cx, cy = (pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2
        wx, wy = pixel_to_world(cx, cy)
        out["center_world"] = [round(wx, 4), round(wy, 4)]
    elif f["type"] == "point":
        x, y = pts[0]
        yaw = float(f.get("yaw", START_YAW))
        out["point_pixel"] = [int(x), int(y)]
        out["yaw"] = round(yaw, 6)
        if f["kind"] == "start_point":
            out["side"] = f.get("side", "IN")
            # OUT: 미편집이면 원본 스폰 정확값 사용, 편집됐으면 픽셀에서 재계산
            sp = f.get("spawn")
            if sp and tuple(dot_from_spawn(*sp)) == (int(x), int(y)):
                ox, oy, oyaw = sp
            else:
                ox, oy, oyaw = spawn_pose_from_dot(x, y, yaw)
            out["spawn_pose"] = {"x": round(ox, 4), "y": round(oy, 4),
                                 "yaw": round(oyaw, 6)}
            wx, wy = pixel_to_world(x, y)
            out["point_world"] = [round(wx, 4), round(wy, 4)]
        else:   # obstacle / traffic_light — 점 자체가 객체 위치
            wd = f.get("world")
            if wd and tuple(world_to_pixel(*wd)) == (int(x), int(y)):
                wx, wy = wd          # 미편집: 원본 정확 월드좌표 보존
            else:
                wx, wy = pixel_to_world(x, y)
            out["point_world"] = [round(wx, 4), round(wy, 4)]
            if "src" in f:
                out["src"] = f["src"]
            out["object_pose"] = {"x": round(wx, 4), "y": round(wy, 4),
                                  "yaw": round(yaw, 6)}
    return out


def feature_from_world(d: dict) -> dict:
    """월드 직렬화 dict → 편집 피처(픽셀)."""
    t = d["type"]
    f = {"kind": d["kind"], "type": t, "id": d["id"]}
    if t == "box":
        f["pts"] = [tuple(world_to_pixel(a, b)) for a, b in d["corners_world"]]
    elif t == "line":
        f["pts"] = [tuple(world_to_pixel(a, b)) for a, b in d["endpoints_world"]]
    elif t == "point":
        # 저장된 point_pixel을 직접 복원 (world 경유 시 1px 라운딩 드리프트 방지)
        if "point_pixel" in d:
            f["pts"] = [tuple(d["point_pixel"])]
        else:
            f["pts"] = [tuple(world_to_pixel(*d["point_world"]))]
        f["yaw"] = float(d.get("yaw", START_YAW))
        if "side" in d:
            f["side"] = d["side"]
        if "src" in d:
            f["src"] = d["src"]
        # 원본 정확값 복원 (재저장 시 정밀도 유지)
        if "spawn_pose" in d:
            sp = d["spawn_pose"]
            f["spawn"] = (sp["x"], sp["y"], sp["yaw"])
        if d.get("kind") in ("obstacle", "traffic_light"):
            f["world"] = tuple(d["point_world"])
    return f


TRACK_PNG_PATH = (
    "/home/autolab/VLA_simulation/src/simulation_pkg/"
    "models/race_track/materials/textures/track.png"
)
DEFAULT_GT_PATH = os.path.expanduser("~/track_gt_manual.json")


# ─── 알고리즘 함수 ────────────────────────────────────────────────────────────
def detect_road_by_color(img: np.ndarray, seed_bgr: np.ndarray, tol: int) -> np.ndarray:
    """BGR 색상 ± tolerance 마스크."""
    diff = np.abs(img.astype(np.int32) - seed_bgr.astype(np.int32)).max(axis=2)
    m = (diff <= tol).astype(np.uint8) * 255
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=3)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=2)
    return m


def detect_road_by_v(img: np.ndarray, v_min: int, v_max: int) -> np.ndarray:
    """HSV 밝기(V) 범위 마스크."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v, s = hsv[:, :, 2], hsv[:, :, 1]
    m = ((v >= v_min) & (v <= v_max) & (s < 35)).astype(np.uint8) * 255
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=3)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=2)
    return m


def find_ring_boundary(img_bgr: np.ndarray, cx: int, cy: int) -> float:
    """
    내부 타원과 도로 링 사이 경계 반지름 자동 검출.

    원리: 내부 타원 외벽의 흰 경계선이 반지름 히스토그램에서
          r=80~320px 범위 내 뚜렷한 피크로 나타남.
          피크 반지름 + 여유(20px) = 도로 링 시작 지점.

    Returns: min_r [px]  (0이면 검출 실패 → 필터 없이 진행)
    """
    H, W = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 2] > 190) & (hsv[:, :, 1] < 30)

    ys_all, xs_all = np.mgrid[0:H, 0:W]
    r_all = np.sqrt((xs_all - cx) ** 2 + (ys_all - cy) ** 2).astype(np.float32)

    r_white = r_all[white]
    if len(r_white) < 200:
        return 0.0

    BIN = 5
    r_max = 400
    hist_w, _ = np.histogram(r_white, bins=np.arange(0, r_max + BIN, BIN))

    # 스무딩 없이 raw 히스토그램 사용 (스무딩 시 피크 위치가 이동해 오검출 발생)
    # r=80~320px 범위에서 가장 높은 피크 = 내부 타원 경계선 흰 줄
    lo = max(0,            80  // BIN)
    hi = min(len(hist_w),  320 // BIN)
    if hi <= lo:
        return 0.0

    peak_bin = lo + int(np.argmax(hist_w[lo:hi]))
    peak_r   = float(peak_bin * BIN)
    peak_val = float(hist_w[peak_bin])

    # 피크 유의성: 범위 평균 대비 1.5배 이상이어야 경계선으로 인정
    bg_mean = float(hist_w[lo:hi].mean())
    if bg_mean < 1 or peak_val < 1.5 * bg_mean:
        return 0.0

    return float(peak_r + 20)   # 피크 + 여유 20px = 도로 링 시작


def split_lanes_polar(road_mask: np.ndarray, n_bins: int = 360,
                      min_r: float = 0.0):
    """
    극좌표 기반 차선 분리.

    min_r > 0 이면 그 반지름 안쪽(내부 타원)을 제외하고 분리.
    → 곡선 구간에서도 정확한 1·2차선 분리 가능.

    반환: (inner_mask, outer_mask, center_xy)
    """
    H, W = road_mask.shape
    M = cv2.moments(road_mask)
    if M['m00'] == 0:
        return None, None, (W // 2, H // 2)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # 2D 반지름 그리드 (모든 픽셀)
    ys_all, xs_all = np.mgrid[0:H, 0:W]
    r_all = np.sqrt((xs_all - cx) ** 2 + (ys_all - cy) ** 2).astype(np.float32)

    # 내부 타원 제외
    if min_r > 0:
        eff = (road_mask > 128) & (r_all > min_r)
    else:
        eff = road_mask > 128

    ys, xs = np.where(eff)
    if len(ys) < 500:          # 너무 적으면 전체 마스크 fallback
        ys, xs = np.where(road_mask > 128)

    dx = xs.astype(float) - cx
    dy = ys.astype(float) - cy
    radii  = np.sqrt(dx ** 2 + dy ** 2)
    angles = np.arctan2(dy, dx)

    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bidx  = np.clip(np.digitize(angles, edges) - 1, 0, n_bins - 1)

    divider = np.zeros(n_bins)
    for b in range(n_bins):
        rb = radii[bidx == b]
        if len(rb) > 0:
            # (min+max)/2 : 링의 반지름 무게 편향을 제거한 중앙 분리선
            # median 사용 시 외곽(큰 r)쪽으로 치우쳐 곡선 구간 오류 발생
            divider[b] = (rb.min() + rb.max()) / 2.0

    # 빈 빈(bin) 선형 보간 (S자 구간에서 데이터 없는 각도)
    valid = divider > 0
    if valid.any() and not valid.all():
        xi = np.arange(n_bins)
        divider = np.interp(xi, xi[valid], divider[valid])

    piv = divider[bidx]
    inner = np.zeros((H, W), np.uint8)
    outer = np.zeros((H, W), np.uint8)
    inner[ys[radii <  piv], xs[radii <  piv]] = 255
    outer[ys[radii >= piv], xs[radii >= piv]] = 255
    return inner, outer, (cx, cy)


def _smooth_closed_curve(pts: list, window: int = 15, iters: int = 4) -> list:
    """
    닫힌 곡선 (원형 트랙)을 주기 경계 조건으로 이동 평균 스무딩.
    S자 곡선 구간의 튀는 점을 인접 점으로 당겨 부드럽게 만든다.
    """
    if len(pts) < window + 1:
        return pts
    arr = np.array(pts, dtype=float)
    n = len(arr)
    hw = window // 2
    for _ in range(iters):
        smooth = np.zeros_like(arr)
        for i in range(n):
            idx = [(i + j - hw) % n for j in range(window)]
            smooth[i] = arr[idx].mean(axis=0)
        arr = smooth
    return [(int(round(x)), int(round(y))) for x, y in arr]


def polar_centerline(mask: np.ndarray, center_xy, n_bins: int = 720,
                     smooth_window: int = 15, smooth_iters: int = 4):
    """
    극좌표 각도별 평균 반지름 → 직교 좌표 중앙선 포인트.

    n_bins=720 (0.5° 단위): 곡선 구간 해상도 향상
    이후 주기적 이동 평균 스무딩으로 S자 곡선 튀는 점 제거.
    """
    cx, cy = center_xy
    H, W = mask.shape
    ys, xs = np.where(mask > 128)
    if len(ys) == 0:
        return []
    dx = xs.astype(float) - cx
    dy = ys.astype(float) - cy
    radii  = np.sqrt(dx ** 2 + dy ** 2)
    angles = np.arctan2(dy, dx)
    edges  = np.linspace(-np.pi, np.pi, n_bins + 1)
    bidx   = np.clip(np.digitize(angles, edges) - 1, 0, n_bins - 1)
    # 각도별 중앙 반지름 계산 ((min+max)/2 : 링 면적 편향 없는 중앙값)
    raw_r = np.full(n_bins, -1.0)
    for b in range(n_bins):
        rb = radii[bidx == b]
        if len(rb) >= 2:
            raw_r[b] = (rb.min() + rb.max()) / 2.0
        elif len(rb) == 1:
            raw_r[b] = rb[0]

    # 빈 빈 선형 보간 (주기 경계)
    valid_mask = raw_r >= 0
    if valid_mask.any() and not valid_mask.all():
        xi = np.arange(n_bins)
        raw_r = np.interp(xi, xi[valid_mask], raw_r[valid_mask])

    pts = []
    for b in range(n_bins):
        if raw_r[b] < 0:
            continue
        theta = (edges[b] + edges[b + 1]) / 2.0
        px_ = int(cx + raw_r[b] * math.cos(theta))
        py_ = int(cy + raw_r[b] * math.sin(theta))
        if 0 <= px_ < W and 0 <= py_ < H:
            pts.append((px_, py_))

    # 닫힌 곡선 스무딩 (S자 곡선 튀는 점 완화)
    if smooth_window > 1 and len(pts) > smooth_window:
        pts = _smooth_closed_curve(pts, window=smooth_window, iters=smooth_iters)

    return pts


def _resample_contour(cnt: np.ndarray, n_pts: int) -> list:
    """컨투어 → 균등 간격 n_pts개 꼭짓점."""
    if len(cnt) < 3:
        return [(int(x), int(y)) for x, y in cnt]
    dv    = np.diff(cnt, axis=0)
    dists = np.sqrt((dv ** 2).sum(axis=1))
    arc   = np.concatenate([[0], np.cumsum(dists)])
    total = arc[-1]
    if total < 1:
        return [(int(x), int(y)) for x, y in cnt[:n_pts]]
    targets = np.linspace(0, total, n_pts, endpoint=False)
    result  = []
    for t in targets:
        i   = max(0, min(int(np.searchsorted(arc, t, side='right')) - 1, len(cnt) - 2))
        seg = arc[i + 1] - arc[i]
        f   = (t - arc[i]) / seg if seg > 1e-6 else 0.0
        pt  = cnt[i] + f * (cnt[i + 1] - cnt[i])
        result.append((int(round(pt[0])), int(round(pt[1]))))
    return result


def _find_enclosed_holes(mask: np.ndarray) -> np.ndarray:
    """
    Flood fill로 외부와 연결된 배경을 제거하고,
    마스크 내부에 완전히 둘러싸인 구멍(hole) 영역만 반환.

    링(도넛) 마스크에서 RETR_CCOMP가 구멍을 놓치는 경우도 확실히 탐지.
    """
    h, w = mask.shape
    inv    = 255 - mask          # 도로=0, 비도로=255
    # 1px 패딩: 코너에서 floodFill 시작이 보장되도록
    padded = cv2.copyMakeBorder(inv, 1, 1, 1, 1,
                                cv2.BORDER_CONSTANT, value=255)
    # 코너(0,0)에서 연결된 모든 255(배경) → 128로 마킹
    cv2.floodFill(padded, None, (0, 0), 128)
    # 남아있는 255 = 외부와 단절된 내부 구멍
    holes = (padded[1:h+1, 1:w+1] == 255).astype(np.uint8) * 255
    return holes


def mask_to_polygons(mask: np.ndarray,
                     n_out: int = 80,
                     n_inn: int = 60) -> tuple:
    """
    마스크 → (외곽 폴리곤, 내곽/구멍 폴리곤).

    - outer: 가장 큰 외곽 컨투어  → fillPoly(255)
    - inner: 내부 enclosed hole   → fillPoly(0, 구멍 뚫기)
    링(도넛) 마스크 전용. 단순 폴리곤이면 inner=[]
    """
    # ── 외곽 경계 ────────────────────────────────────────────────────────────
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return [], []
    cnt_out = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt_out) < 200:
        return [], []
    outer_pts = _resample_contour(cnt_out.reshape(-1, 2), n_out)

    # ── 내곽 구멍 (flood fill 방식으로 확실하게 탐지) ─────────────────────────
    holes = _find_enclosed_holes(mask)
    inner_pts = []
    if holes.any():
        hole_cnts, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
        if hole_cnts:
            cnt_inn = max(hole_cnts, key=cv2.contourArea)
            if cv2.contourArea(cnt_inn) > 200:
                inner_pts = _resample_contour(cnt_inn.reshape(-1, 2), n_inn)

    return outer_pts, inner_pts


def _poly_centroid(pts: list) -> tuple:
    """폴리곤 꼭짓점 리스트의 무게중심."""
    arr = np.array(pts, dtype=float)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def _ray_poly_t(ox: float, oy: float, dx: float, dy: float,
                poly: list) -> float | None:
    """
    레이 (ox,oy) + t*(dx,dy) 와 닫힌 폴리곤의 첫 교점 파라미터 t.
    없으면 None.
    Cramer 공식으로 각 엣지와의 교점을 계산.
    """
    n = len(poly)
    best_t = None
    for i in range(n):
        ax, ay = float(poly[i][0]),         float(poly[i][1])
        bx, by = float(poly[(i+1) % n][0]), float(poly[(i+1) % n][1])
        ex, ey = bx - ax, by - ay
        # det([[dx, -ex], [dy, -ey]])  → -dx*ey + ex*dy
        det = -dx * ey + ex * dy
        if abs(det) < 1e-9:
            continue
        rx, ry = ax - ox, ay - oy
        t = (-ey * rx + ex * ry) / det
        s = (-dy * rx + dx * ry) / det
        if t > 1e-9 and -1e-9 <= s <= 1.0 + 1e-9:
            if best_t is None or t < best_t:
                best_t = t
    return best_t


def _arc_resample_np(pts: list, n: int) -> np.ndarray:
    """
    폴리곤 꼭짓점 리스트 → 호 길이 기준 균등 n개 리샘플 (닫힌 루프).
    반환: (n, 2) float64 ndarray
    """
    arr = np.array(pts, dtype=float)
    n_orig = len(arr)
    arr_c = np.vstack([arr, arr[[0]]])          # 닫힌 루프
    diffs = np.diff(arr_c, axis=0)
    segs  = np.sqrt((diffs ** 2).sum(axis=1))
    arc   = np.concatenate([[0.0], np.cumsum(segs)])
    total = arc[-1]
    if total < 1e-9:
        return np.tile(arr[0], (n, 1))
    targets = np.linspace(0.0, total, n, endpoint=False)
    result  = np.zeros((n, 2))
    for k, t in enumerate(targets):
        i = max(0, min(int(np.searchsorted(arc[:-1], t, 'right')) - 1, n_orig - 1))
        j = (i + 1) % n_orig
        seg = arc[i + 1] - arc[i]
        f   = (t - arc[i]) / seg if seg > 1e-9 else 0.0
        result[k] = arr[i] + f * (arr[j] - arr[i])
    return result


def _signed_area_arr(arr: np.ndarray) -> float:
    """Shoelace 공식으로 부호 면적 계산 (양수 = CCW)."""
    x, y = arr[:, 0], arr[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def centerline_from_polygon_midpoint(p_out: list, p_inn: list,
                                      n_pts: int = 720,
                                      smooth_window: int = 15,
                                      smooth_iters: int = 4) -> list:
    """
    각 외곽 폴리곤 점에서 내곽 폴리곤의 실제 최근접 점 → 중점.

    코너 편향 해결:
      - 호 길이 인덱스 대응(out[i]↔inn[i])은 외곽/내곽 둘레 길이가 달라
        코너에서 다른 트랙 위치를 가리키는 문제 발생.
      - 최근접 점 방식은 방향 편향 없이 기하학적으로 가장 가까운 내곽 경계점.
      - 내곽을 4배 밀도(4×n_pts)로 리샘플해 최근접 점 정확도 향상.

    동작 순서:
      1. p_out → n_pts 호 길이 균등 리샘플 (출력 순서 보장)
      2. p_inn → 4×n_pts 고밀도 리샘플 (최근접 검색 정확도)
      3. 각 out[i] → inn_dense에서 argmin(거리) → nearest[i]
      4. mid[i] = (out[i] + nearest[i]) / 2
    """
    if not p_out or not p_inn:
        return []

    out_r     = _arc_resample_np(p_out, n_pts)        # (n_pts, 2)
    inn_dense = _arc_resample_np(p_inn, n_pts * 4)    # (4*n_pts, 2) 고밀도

    # ── 벡터화 최근접 점 탐색 ─────────────────────────────────────────────────
    # diff: (n_pts, 4*n_pts, 2)  →  dist_sq: (n_pts, 4*n_pts)
    diff     = out_r[:, np.newaxis, :] - inn_dense[np.newaxis, :, :]
    dist_sq  = (diff ** 2).sum(axis=2)
    nearest  = inn_dense[dist_sq.argmin(axis=1)]       # (n_pts, 2)

    mid = (out_r + nearest) / 2.0
    pts = [(int(round(float(x))), int(round(float(y)))) for x, y in mid]

    if smooth_window > 1 and len(pts) > smooth_window:
        pts = _smooth_closed_curve(pts, smooth_window, smooth_iters)

    return pts


# ── 레이 캐스팅 방식 (참고용, extract_from_masks에서는 midpoint 방식 사용) ──
def centerline_from_polygon_pair(p_out: list, p_inn: list, center_xy,
                                  n_pts: int = 720,
                                  smooth_window: int = 15,
                                  smooth_iters: int = 4) -> list:
    """레이 캐스팅 교점 중점 — 코너 편향 있음, 참고용으로 보존."""
    if not p_out or not p_inn:
        return []
    cx, cy = float(center_xy[0]), float(center_xy[1])
    pts, miss = [], 0
    for i in range(n_pts):
        angle = -math.pi + 2.0 * math.pi * i / n_pts
        dx, dy = math.cos(angle), math.sin(angle)
        t_out = _ray_poly_t(cx, cy, dx, dy, p_out)
        t_inn = _ray_poly_t(cx, cy, dx, dy, p_inn)
        if t_out is None or t_inn is None:
            miss += 1
            continue
        t_mid = (t_out + t_inn) / 2.0
        pts.append((int(round(cx + t_mid * dx)), int(round(cy + t_mid * dy))))
    if smooth_window > 1 and len(pts) > smooth_window:
        pts = _smooth_closed_curve(pts, smooth_window, smooth_iters)
    return pts


def centerline_from_dist_transform(mask: np.ndarray, center_xy,
                                   n_pts: int = 720,
                                   smooth_window: int = 15,
                                   smooth_iters: int = 4) -> list:
    """
    거리 변환(distance transform) 기반 차선 중앙선 추출.

    polar_centerline 대비 곡선 구간 정확도 향상.

    원리:
      - 각 마스크 픽셀 → 가장 가까운 경계까지의 거리 계산
      - 각도 빈(n_pts)마다 '거리 최대' 픽셀 선택 = 실제 차선 중앙
      - polar (min+max)/2 방식은 반지름 기준이라 곡선에서 치우침 발생

    Returns: [(px, py), ...]  극좌표 각도 순 정렬 + 스무딩
    """
    if not (mask > 0).any():
        return []

    # ── 거리 변환 ─────────────────────────────────────────────────────────────
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_d = float(dist.max())
    if max_d < 2:
        return []

    cx, cy = center_xy
    H, W   = mask.shape

    ys_all, xs_all = np.where(mask > 0)
    if len(ys_all) == 0:
        return []

    dx        = xs_all.astype(float) - cx
    dy        = ys_all.astype(float) - cy
    angles    = np.arctan2(dy, dx)
    dist_vals = dist[ys_all, xs_all]

    # ── 각도 빈별 거리 최대 픽셀 선택 ─────────────────────────────────────────
    edges = np.linspace(-np.pi, np.pi, n_pts + 1)
    bidx  = np.clip(np.digitize(angles, edges) - 1, 0, n_pts - 1)

    raw_x = np.full(n_pts, -1.0)
    raw_y = np.full(n_pts, -1.0)

    for b in range(n_pts):
        sel = bidx == b
        if not sel.any():
            continue
        best     = int(np.argmax(dist_vals[sel]))
        raw_x[b] = float(xs_all[sel][best])
        raw_y[b] = float(ys_all[sel][best])

    # ── 빈 빈 보간 ────────────────────────────────────────────────────────────
    valid = raw_x >= 0
    if not valid.any():
        return []
    if not valid.all():
        xi    = np.arange(n_pts)
        raw_x = np.interp(xi, xi[valid], raw_x[valid])
        raw_y = np.interp(xi, xi[valid], raw_y[valid])

    pts = [(int(round(raw_x[b])), int(round(raw_y[b])))
           for b in range(n_pts)
           if 0 <= int(round(raw_x[b])) < W and 0 <= int(round(raw_y[b])) < H]

    # ── 스무딩 ────────────────────────────────────────────────────────────────
    if smooth_window > 1 and len(pts) > smooth_window:
        pts = _smooth_closed_curve(pts, smooth_window, smooth_iters)

    return pts


# ─── 백그라운드 워커 ──────────────────────────────────────────────────────────
class Worker(QThread):
    """
    mode:
      "color"  : BGR 피커로 도로 마스크 → 차선 분리
      "v_range": V 범위로 도로 마스크 → 차선 분리
    항상 차선 분리까지 한 번에 수행.
    """
    done = Signal(dict)

    def __init__(self, img, mode,
                 seed_bgr=None, tol=25,
                 v_min=100, v_max=150,
                 n_bins=720, parent=None):
        super().__init__(parent)
        self._img    = img.copy()
        self._mode   = mode
        self._seed   = seed_bgr
        self._tol    = tol
        self._vmin   = v_min
        self._vmax   = v_max
        self._bins   = n_bins

    def run(self):
        img = self._img
        # ── 1단계: 도로 마스크 ───────────────────────────────────────────────
        if self._mode == "color" and self._seed is not None:
            road = detect_road_by_color(img, self._seed, self._tol)
        else:
            road = detect_road_by_v(img, self._vmin, self._vmax)

        n_road = int((road > 128).sum())
        if n_road < 1000:
            self.done.emit({"ok": False,
                            "msg": f"도로 픽셀이 너무 적습니다 ({n_road}px). "
                                   "다른 위치를 클릭하거나 허용 오차를 늘려보세요."})
            return

        # ── 2단계: 내부 타원 경계 자동 검출 ──────────────────────────────────
        # 내부 타원(주차장)이 도로와 동일 색 → 솔리드 디스크 형태가 됨
        # 무게중심 계산
        M = cv2.moments(road)
        if M['m00'] == 0:
            self.done.emit({"ok": False, "msg": "무게중심 계산 실패"})
            return
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # ★ 경계 검출에는 항상 V-range 회색 도로 무게중심 사용 (색상 피커가 흰 줄/배경을
        #   샘플한 경우에도 올바른 내부 타원 경계를 찾기 위함)
        gray_road = detect_road_by_v(img, 95, 155)
        M_gray = cv2.moments(gray_road)
        if M_gray['m00'] > 0:
            cx_ref = int(M_gray['m10'] / M_gray['m00'])
            cy_ref = int(M_gray['m01'] / M_gray['m00'])
        else:
            cx_ref, cy_ref = cx, cy  # fallback

        boundary_r = find_ring_boundary(img, cx_ref, cy_ref)
        # 범위 안전 처리
        if not (50 < boundary_r < 400):
            boundary_r = 0.0

        # ── 3단계: 차선 분리 (내부 타원 제외 후 극좌표 분리) ─────────────────
        inner, outer, center = split_lanes_polar(road, self._bins,
                                                 min_r=boundary_r)
        if inner is None:
            self.done.emit({"ok": False, "msg": "차선 분리 실패"})
            return

        # ── 4단계: 중앙선 추출 (720 빈 + 스무딩으로 곡선 품질 향상) ──────────
        pts_i = polar_centerline(inner, center, n_bins=720,
                                 smooth_window=15, smooth_iters=4)
        pts_o = polar_centerline(outer, center, n_bins=720,
                                 smooth_window=15, smooth_iters=4)

        self.done.emit({
            "ok":         True,
            "road":       road,
            "inner":      inner,
            "outer":      outer,
            "pts_i":      pts_i,
            "pts_o":      pts_o,
            "center":     center,
            "n_road":     n_road,
            "boundary_r": boundary_r,
        })


# ─── 캔버스 ───────────────────────────────────────────────────────────────────
class Canvas(QLabel):
    hoverMoved  = Signal(int, int)
    colorPicked = Signal(int, int, int, int, int)   # px,py,B,G,R
    pointAdded  = Signal()

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

        self._src: np.ndarray | None = None
        self._base: QPixmap | None   = None
        self._scale = 0.75

        # 오버레이
        self._road:  np.ndarray | None = None
        self._inner: np.ndarray | None = None
        self._outer: np.ndarray | None = None
        self._center = None
        self._pts_i: list = []
        self._pts_o: list = []
        self._manual: list = []

        self._show_lane = "both"   # "both" | "inner" | "outer"
        self._show_landmarks = True
        self._show_mask = True
        self._mode  = "draw"
        self._brush = 10
        self._drag  = False
        self._hover = QPoint(0, 0)
        self._box_start: QPoint | None = None   # box_erase 드래그 시작점

        # ── 피처 주석 레이어 (주차칸/IN·OUT선/횡단보도/출발점) ──────────────────
        self._feats: list = []                  # feature dict 리스트 (픽셀 기하)
        self._show_feats = True
        self._feat_drag = None                  # (feat_idx, handle_idx) | (idx,'move')
        self._feat_anchor = (0, 0)              # 'move' 드래그 기준점
        self._feat_hover = None                 # (feat_idx, handle_idx)
        self._feat_add_start = False            # True면 클릭으로 출발점 추가
        self._feat_add_side = "IN"              # 추가할 출발점 측 (IN/OUT)

        # ── 수동 마스크 레이어 ──────────────────────────────────────────────────
        self._mask_i: np.ndarray | None = None   # 1차선 수동 마스크
        self._mask_o: np.ndarray | None = None   # 2차선 수동 마스크
        self._active_lane = "inner"              # 현재 편집 차선
        self._poly_fill_val = 255               # 255=채우기, 0=지우기

        # ── 폴리곤 그리기 상태 ────────────────────────────────────────────────────
        self._poly_pts: list = []               # 진행 중인 폴리곤 꼭짓점
        self._mask_i_hist: list = []            # undo 히스토리 (최대 20)
        self._mask_o_hist: list = []

        # ── 편집 폴리곤 (마스크 꼭짓점 드래그 편집) ─────────────────────────────
        # 링 마스크는 외곽(out) + 구멍(inn) 두 폴리곤으로 표현
        self._edit_poly_i_out: list = []        # 1차선 외곽 폴리곤
        self._edit_poly_i_inn: list = []        # 1차선 내곽(구멍) 폴리곤
        self._edit_poly_o_out: list = []        # 2차선 외곽 폴리곤
        self._edit_poly_o_inn: list = []        # 2차선 내곽(구멍) 폴리곤
        self._edit_active = "inner"             # 현재 편집 차선
        self._edit_boundary = "outer"           # 현재 편집 경계 ("outer"|"inner")
        self._edit_drag_idx: int = -1           # 드래그 중인 꼭짓점
        self._edit_hover_idx: int = -1          # 커서 근처 꼭짓점

        self.setFocusPolicy(Qt.ClickFocus)      # 키보드 이벤트 수신

    # ── API ───────────────────────────────────────────────────────────────────
    def load(self, path: str) -> bool:
        img = cv2.imread(path)
        if img is None: return False
        self._src  = img
        self._base = self._to_pixmap(img)
        self._road = self._inner = self._outer = None
        self._center = None; self._pts_i = []; self._pts_o = []; self._manual = []
        # 이미지 크기에 맞춰 수동 마스크 초기화
        h, w = img.shape[:2]
        self._mask_i = np.zeros((h, w), np.uint8)
        self._mask_o = np.zeros((h, w), np.uint8)
        self._mask_i_hist = []; self._mask_o_hist = []
        self._poly_pts = []
        self._edit_poly_i_out = []; self._edit_poly_i_inn = []
        self._edit_poly_o_out = []; self._edit_poly_o_inn = []
        self._edit_drag_idx = -1; self._edit_hover_idx = -1
        self._feat_drag = None; self._feat_hover = None
        self._refresh(); return True

    def set_result(self, road, inner, outer, pts_i, pts_o, center):
        self._road = road; self._inner = inner; self._outer = outer
        self._pts_i = pts_i; self._pts_o = pts_o; self._center = center
        self._refresh()

    def clear_result(self):
        self._road = self._inner = self._outer = None
        self._center = None; self._pts_i = []; self._pts_o = []
        self._refresh()

    def set_scale(self, s):  self._scale = max(0.1, min(5.0, s)); self._refresh()
    def set_mode(self, m):
        if m != self._mode:
            self._poly_pts = []   # 모드 바뀌면 진행 중인 폴리곤 초기화
        self._mode = m
        self._box_start = None
        self.setCursor({
            "draw":       Qt.CrossCursor,
            "erase":      Qt.PointingHandCursor,
            "picker":     Qt.WhatsThisCursor,
            "box_erase":  Qt.SizeAllCursor,
            "mask_poly":  Qt.CrossCursor,
            "poly_edit":  Qt.ArrowCursor,
            "feature":    Qt.ArrowCursor,
        }.get(m, Qt.CrossCursor))

    def set_active_lane(self, lane: str):
        """현재 마스크 칠하기 대상 차선 설정 ('inner' | 'outer')."""
        self._active_lane = lane

    def clear_mask(self, lane: str | None = None):
        """마스크 초기화. lane=None이면 둘 다."""
        if self._mask_i is None: return
        h, w = self._mask_i.shape
        if lane in (None, "inner"):
            self._mask_i = np.zeros((h, w), np.uint8)
        if lane in (None, "outer"):
            self._mask_o = np.zeros((h, w), np.uint8)
        self._refresh()

    def import_auto_masks(self) -> bool:
        """자동 검출 마스크(_inner/_outer) → 수동 마스크(_mask_i/_mask_o) 복사."""
        if self._inner is None or self._outer is None:
            return False
        self._mask_i = self._inner.copy()
        self._mask_o = self._outer.copy()
        self._mask_i_hist = []
        self._mask_o_hist = []
        self._poly_pts = []
        self._refresh()
        return True

    def extract_from_masks(self):
        """
        수동 마스크 / 편집 폴리곤 → 중앙선 추출. (ni, no) 반환.

        우선순위:
          1. 편집 폴리곤 쌍(outer+inner) 있으면 → centerline_from_polygon_pair
             (레이 캐스팅 방식: 폴리곤 경계 교점 중점)
          2. 폴리곤 없으면 → centerline_from_dist_transform (마스크 기반)
        """
        if self._mask_i is None or self._mask_o is None:
            return 0, 0

        # ── 레이 캐스팅 기준 중심점 계산 ───────────────────────────────────────
        # 1순위: inner_inn (트랙 중심 섬 경계) 무게중심
        # 2순위: 기존 center
        # 3순위: 두 마스크 합산 무게중심
        center = self._center
        ref_poly = (self._edit_poly_i_inn or self._edit_poly_o_inn
                    or self._edit_poly_i_out or self._edit_poly_o_out)
        if ref_poly:
            center = _poly_centroid(ref_poly)   # 항상 트랙 중심에 가장 가까운 폴리곤
        if center is None:
            combined = cv2.bitwise_or(self._mask_i, self._mask_o)
            M = cv2.moments(combined)
            if M['m00'] > 0:
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            elif self._src is not None:
                h, w = self._src.shape[:2]
                center = (w // 2, h // 2)
            else:
                return 0, 0

        pts_i, pts_o = [], []
        method_i = method_o = "?"

        # ── 1차선 중앙선 ────────────────────────────────────────────────────────
        if self._edit_poly_i_out and self._edit_poly_i_inn:
            pts_i = centerline_from_polygon_midpoint(
                self._edit_poly_i_out, self._edit_poly_i_inn,
                n_pts=720, smooth_window=15, smooth_iters=4)
            method_i = "최근접 중점"
        elif (self._mask_i > 0).any():
            pts_i = centerline_from_dist_transform(
                self._mask_i, center, n_pts=720, smooth_window=15, smooth_iters=4)
            method_i = "거리변환"

        # ── 2차선 중앙선 ────────────────────────────────────────────────────────
        if self._edit_poly_o_out and self._edit_poly_o_inn:
            pts_o = centerline_from_polygon_midpoint(
                self._edit_poly_o_out, self._edit_poly_o_inn,
                n_pts=720, smooth_window=15, smooth_iters=4)
            method_o = "최근접 중점"
        elif (self._mask_o > 0).any():
            pts_o = centerline_from_dist_transform(
                self._mask_o, center, n_pts=720, smooth_window=15, smooth_iters=4)
            method_o = "거리변환"

        self._pts_i   = pts_i
        self._pts_o   = pts_o
        self._center  = center
        self._method_i = method_i
        self._method_o = method_o
        self._refresh()
        return len(pts_i), len(pts_o)

    # ── 피처 주석 ─────────────────────────────────────────────────────────────
    def load_features(self, feats: list):
        self._feats = [dict(f, pts=[tuple(p) for p in f["pts"]]) for f in feats]
        self._feat_drag = None; self._feat_hover = None
        self._refresh()

    def get_features(self) -> list:
        return self._feats

    def clear_features(self):
        self._feats = []; self._feat_drag = None; self._feat_hover = None
        self._refresh()

    def toggle_features(self, v: bool):
        self._show_feats = v; self._refresh()

    def set_feat_add_start(self, v: bool):
        self._feat_add_start = v

    def set_feat_add_side(self, side: str):
        self._feat_add_side = side

    def add_start_point(self, px: int, py: int, side: str = "IN"):
        yaw = START_YAW if side == "IN" else math.pi
        n = sum(1 for f in self._feats
                if f["kind"] == "start_point" and f.get("side") == side)
        self._feats.append({"kind": "start_point", "type": "point",
                            "id": f"{side}{n+1}", "side": side,
                            "pts": [(px, py)], "yaw": yaw})
        self.pointAdded.emit(); self._refresh()

    def _renumber_starts(self):
        cnt = {}
        for f in self._feats:
            if f["kind"] == "start_point":
                s = f.get("side", "IN")
                cnt[s] = cnt.get(s, 0) + 1
                f["id"] = f"{s}{cnt[s]}"

    def delete_feature_at(self, px: int, py: int) -> bool:
        """커서 근처 출발점 삭제 (출발점만 삭제 가능)."""
        thr = 14 / self._scale
        for i, f in enumerate(self._feats):
            if f["kind"] != "start_point":
                continue
            vx, vy = f["pts"][0]
            if math.hypot(px - vx, py - vy) < thr:
                self._feats.pop(i); self._renumber_starts()
                self._feat_drag = None; self._feat_hover = None
                self.pointAdded.emit(); self._refresh(); return True
        return False

    @staticmethod
    def _pt_in_poly(px, py, poly) -> bool:
        inside = False; n = len(poly)
        for i in range(n):
            ax, ay = poly[i]; bx, by = poly[(i + 1) % n]
            if (ay > py) != (by > py):
                xint = ax + (py - ay) * (bx - ax) / (by - ay + 1e-12)
                if px < xint:
                    inside = not inside
        return inside

    @staticmethod
    def _dist_seg(px, py, a, b) -> float:
        ax, ay = a; bx, by = b
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 < 1e-9:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

    def _feat_hit(self, px, py):
        """반환: ('handle', fi, hi) | ('move', fi, None) | None."""
        thr = 12 / self._scale
        best = None; bestd = thr
        for fi, f in enumerate(self._feats):
            for hi, (vx, vy) in enumerate(f["pts"]):
                d = math.hypot(px - vx, py - vy)
                if d < bestd:
                    best = ("handle", fi, hi); bestd = d
        if best:
            return best
        # 핸들 미접촉 → 박스 내부/선 위 = 전체 이동
        for fi, f in enumerate(self._feats):
            if f["type"] == "box" and self._pt_in_poly(px, py, f["pts"]):
                return ("move", fi, None)
            if f["type"] == "line" and self._dist_seg(px, py, *f["pts"]) < thr:
                return ("move", fi, None)
        return None

    # ── 폴리곤 마스크 ─────────────────────────────────────────────────────────
    def _close_polygon(self):
        """현재 폴리곤을 닫고 활성 마스크에 채움."""
        if len(self._poly_pts) < 3:
            self._poly_pts = []; self._refresh(); return
        mask = self._mask_i if self._active_lane == "inner" else self._mask_o
        hist = self._mask_i_hist if self._active_lane == "inner" else self._mask_o_hist
        if mask is None: self._poly_pts = []; return
        # undo 히스토리 저장 (최대 20)
        hist.append(mask.copy())
        if len(hist) > 20: hist.pop(0)
        pts_arr = np.array(self._poly_pts, dtype=np.int32)
        cv2.fillPoly(mask, [pts_arr], self._poly_fill_val)
        self._poly_pts = []
        self.pointAdded.emit()
        self._refresh()

    def undo_polygon(self):
        """마지막 폴리곤 취소 (히스토리 복원)."""
        if self._poly_pts:   # 진행 중인 폴리곤이면 취소
            self._poly_pts = []; self._refresh(); return True
        hist = self._mask_i_hist if self._active_lane == "inner" else self._mask_o_hist
        if not hist: return False
        if self._active_lane == "inner":
            self._mask_i = hist.pop()
        else:
            self._mask_o = hist.pop()
        self._refresh(); return True

    def cancel_polygon(self):
        """진행 중인 폴리곤 취소."""
        self._poly_pts = []; self._refresh()

    # ── 편집 폴리곤 (꼭짓점 드래그) ──────────────────────────────────────────
    def set_edit_lane(self, lane: str):
        self._edit_active = lane
        self._edit_drag_idx = -1; self._edit_hover_idx = -1
        self._refresh()

    def set_edit_boundary(self, boundary: str):
        """'outer' 또는 'inner(구멍)' 경계 선택."""
        self._edit_boundary = boundary
        self._edit_drag_idx = -1; self._edit_hover_idx = -1
        self._refresh()

    def _get_active_edit_poly(self) -> list:
        if self._edit_active == "inner":
            return self._edit_poly_i_out if self._edit_boundary == "outer" \
                   else self._edit_poly_i_inn
        else:
            return self._edit_poly_o_out if self._edit_boundary == "outer" \
                   else self._edit_poly_o_inn

    def _set_active_edit_poly(self, pts: list):
        if self._edit_active == "inner":
            if self._edit_boundary == "outer": self._edit_poly_i_out = pts
            else:                              self._edit_poly_i_inn = pts
        else:
            if self._edit_boundary == "outer": self._edit_poly_o_out = pts
            else:                              self._edit_poly_o_inn = pts

    def _edit_poly_to_mask(self, lane: str):
        """편집 폴리곤(outer + inner hole) → 마스크에 즉시 반영."""
        if lane == "inner":
            p_out, p_inn = self._edit_poly_i_out, self._edit_poly_i_inn
            mask = self._mask_i
        else:
            p_out, p_inn = self._edit_poly_o_out, self._edit_poly_o_inn
            mask = self._mask_o
        if not p_out or mask is None: return
        mask[:] = 0
        cv2.fillPoly(mask, [np.array(p_out, dtype=np.int32)], 255)
        if p_inn:   # 구멍(inner hole) 잘라내기
            cv2.fillPoly(mask, [np.array(p_inn, dtype=np.int32)], 0)

    def import_mask_as_polygon(self, lane: str,
                               n_out: int = 80, n_inn: int = 60) -> bool:
        """현재 마스크 외곽선 → 외곽+내곽 편집 폴리곤으로 변환."""
        mask = self._mask_i if lane == "inner" else self._mask_o
        if mask is None or not (mask > 0).any(): return False
        p_out, p_inn = mask_to_polygons(mask, n_out, n_inn)
        if not p_out: return False
        if lane == "inner":
            self._edit_poly_i_out = p_out
            self._edit_poly_i_inn = p_inn
        else:
            self._edit_poly_o_out = p_out
            self._edit_poly_o_inn = p_inn
        self._refresh(); return True

    def _find_near_vertex(self, px: int, py: int,
                          thresh_screen: float = 12.0) -> int:
        """화면 기준 thresh 픽셀 이내 꼭짓점 인덱스. 없으면 -1."""
        poly = self._get_active_edit_poly()
        thresh_img = thresh_screen / self._scale
        best_i, best_d = -1, thresh_img
        for i, (vx, vy) in enumerate(poly):
            d = math.hypot(px - vx, py - vy)
            if d < best_d: best_i, best_d = i, d
        return best_i

    def delete_near_vertex(self, px: int, py: int) -> bool:
        """커서 근처 꼭짓점 삭제."""
        idx = self._find_near_vertex(px, py, 20.0)
        if idx < 0: return False
        poly = self._get_active_edit_poly()
        if len(poly) <= 3: return False  # 최소 삼각형 유지
        poly.pop(idx)
        self._set_active_edit_poly(poly)
        self._edit_poly_to_mask(self._edit_active)
        self._edit_drag_idx = -1; self._edit_hover_idx = -1
        self._refresh(); return True
    def set_brush(self, b): self._brush = b
    def toggle_landmarks(self, v): self._show_landmarks = v; self._refresh()
    def toggle_mask(self, v):      self._show_mask = v;      self._refresh()
    def set_show_lane(self, v):    self._show_lane = v;      self._refresh()

    def get_manual(self): return list(self._manual)
    def set_manual(self, pts): self._manual = list(pts); self._refresh()
    def clear_manual(self): self._manual = []; self._refresh()

    def get_pts(self, lane):
        """내보낼 포인트 (수동 + 선택 차선 자동)."""
        auto = self._pts_i if lane == "inner" else self._pts_o
        return list(self._manual) + list(auto)

    # ── 렌더링 ────────────────────────────────────────────────────────────────
    def _refresh(self):
        if self._base is None: return
        sw = self._scale
        W  = int(self._base.width()  * sw)
        H  = int(self._base.height() * sw)
        pm = self._base.scaled(W, H, Qt.KeepAspectRatio, Qt.SmoothTransformation).copy()
        p  = QPainter(pm); p.setRenderHint(QPainter.Antialiasing)

        # 수동 마스크 오버레이 (항상 표시, 빨강=1차선 / 파랑=2차선)
        if self._mask_i is not None and (self._mask_i > 0).any():
            self._overlay(p, self._mask_i, QColor(255, 60, 60, 130), sw)
        if self._mask_o is not None and (self._mask_o > 0).any():
            self._overlay(p, self._mask_o, QColor(60, 80, 255, 130), sw)

        if self._show_mask:
            # 자동 검출 차선 마스크 오버레이
            if self._inner is not None and self._outer is not None:
                if self._show_lane in ("inner", "both"):
                    self._overlay(p, self._inner, QColor(220, 60, 60, 80), sw)
                if self._show_lane in ("outer", "both"):
                    self._overlay(p, self._outer, QColor(60, 60, 220, 80), sw)
            elif self._road is not None:
                self._overlay(p, self._road, QColor(0, 180, 255, 70), sw)

        # ── 편집 폴리곤 렌더링 ────────────────────────────────────────────────
        def _draw_edit_poly(poly, color: QColor,
                            is_active_lane: bool, is_active_boundary: bool,
                            dash: bool = False):
            """
            poly: 꼭짓점 리스트
            is_active_lane: 현재 편집 차선 여부
            is_active_boundary: 현재 편집 경계(outer/inner) 여부
            dash: True이면 점선(내곽 구멍 표시)
            """
            if not poly: return
            scr  = [_QPoint(int(x*sw), int(y*sw)) for x, y in poly]
            qpoly = QPolygon(scr)
            # 선 스타일: 활성=실선, 비활성=점선
            pen_style = Qt.DashLine if dash else Qt.SolidLine
            pen_w     = 1.8 if is_active_boundary else 1.0
            pen_col   = color if is_active_lane else \
                        QColor(color.red(), color.green(), color.blue(), 120)
            p.setPen(QPen(pen_col, pen_w, pen_style))
            p.setBrush(Qt.NoBrush)
            p.drawPolygon(qpoly)
            # 꼭짓점: 현재 편집 중인 경계만 표시
            if is_active_lane and is_active_boundary:
                p.setPen(Qt.NoPen)
                for i, pt in enumerate(scr):
                    sx, sy = pt.x(), pt.y()
                    if i == self._edit_hover_idx:
                        p.setBrush(QBrush(QColor(255, 255, 0, 240)))
                        p.drawEllipse(sx-5, sy-5, 10, 10)
                    elif i == self._edit_drag_idx:
                        p.setBrush(QBrush(QColor(255, 140, 0, 240)))
                        p.drawEllipse(sx-5, sy-5, 10, 10)
                    else:
                        p.setBrush(QBrush(QColor(
                            color.red(), color.green(), color.blue(), 200)))
                        p.drawEllipse(sx-3, sy-3, 6, 6)

        lane_i_active = (self._edit_active == "inner")
        lane_o_active = not lane_i_active
        out_active = (self._edit_boundary == "outer")
        inn_active = not out_active

        ci = QColor(255, 100, 100)   # 1차선 색
        co = QColor(100, 130, 255)   # 2차선 색

        # 1차선: 외곽(실선) + 내곽구멍(점선)
        _draw_edit_poly(self._edit_poly_i_out, ci,
                        lane_i_active, lane_i_active and out_active, dash=False)
        _draw_edit_poly(self._edit_poly_i_inn, ci,
                        lane_i_active, lane_i_active and inn_active, dash=True)
        # 2차선: 외곽(실선) + 내곽구멍(점선)
        _draw_edit_poly(self._edit_poly_o_out, co,
                        lane_o_active, lane_o_active and out_active, dash=False)
        _draw_edit_poly(self._edit_poly_o_inn, co,
                        lane_o_active, lane_o_active and inn_active, dash=True)

        # 1차선 중앙선 (청록)
        if self._pts_i and self._show_lane in ("inner", "both"):
            p.setPen(Qt.NoPen); p.setBrush(QBrush(QColor(0, 240, 180, 230)))
            for px_, py_ in self._pts_i:
                p.drawEllipse(int(px_*sw)-3, int(py_*sw)-3, 6, 6)

        # 2차선 중앙선 (주황)
        if self._pts_o and self._show_lane in ("outer", "both"):
            p.setPen(Qt.NoPen); p.setBrush(QBrush(QColor(255, 160, 0, 230)))
            for px_, py_ in self._pts_o:
                p.drawEllipse(int(px_*sw)-3, int(py_*sw)-3, 6, 6)

        # 트랙 중심 (초록 점)
        if self._center:
            cx, cy = self._center
            p.setPen(QPen(QColor(0, 255, 0), 2)); p.setBrush(QBrush(QColor(0, 255, 0)))
            p.drawEllipse(int(cx*sw)-5, int(cy*sw)-5, 10, 10)

        # 수동 점 + 선
        if self._manual:
            if len(self._manual) > 1:
                p.setPen(QPen(QColor(255, 80, 80, 200), max(1, sw*1.2)))
                for i in range(1, len(self._manual)):
                    p.drawLine(int(self._manual[i-1][0]*sw), int(self._manual[i-1][1]*sw),
                               int(self._manual[i][0]*sw),   int(self._manual[i][1]*sw))
            p.setPen(Qt.NoPen); p.setBrush(QBrush(QColor(255, 50, 50, 230)))
            r = max(3, int(4*sw))
            for px_, py_ in self._manual:
                p.drawEllipse(int(px_*sw)-r, int(py_*sw)-r, r*2, r*2)

        # 랜드마크
        if self._show_landmarks:
            for name, (wx_, wy_) in LANDMARKS.items():
                px_, py_ = world_to_pixel(wx_, wy_)
                cx_, cy_ = int(px_*sw), int(py_*sw)
                rad = max(7, int(9*sw))
                p.setPen(QPen(QColor(0, 255, 100), max(2, sw*1.5)))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(cx_-rad, cy_-rad, rad*2, rad*2)
                p.setPen(QPen(QColor(100, 255, 150)))
                p.setFont(QFont("Arial", max(8, int(9*sw))))
                p.drawText(cx_+rad+2, cy_+4, name)

        # ── 피처 주석 (주차칸 / IN·OUT선 / 횡단보도 / 출발점) ─────────────────────
        if self._show_feats and self._feats:
            edit = (self._mode == "feature")
            p.setFont(QFont("Arial", max(8, int(9*sw))))

            def _handle(x, y, col, hot):
                p.setPen(Qt.NoPen)
                if hot:
                    p.setBrush(QBrush(QColor(255, 255, 0, 240)))
                    p.drawEllipse(int(x*sw)-5, int(y*sw)-5, 10, 10)
                else:
                    p.setBrush(QBrush(QColor(col.red(), col.green(), col.blue(), 220)))
                    p.drawEllipse(int(x*sw)-3, int(y*sw)-3, 6, 6)

            for fi, f in enumerate(self._feats):
                rr, gg, bb = FEAT_COLORS.get(f["kind"], (255, 255, 255))
                col = QColor(rr, gg, bb)
                pts = f["pts"]
                if f["type"] == "box":
                    qpoly = QPolygon([_QPoint(int(x*sw), int(y*sw)) for x, y in pts])
                    p.setPen(QPen(col, 2)); p.setBrush(QBrush(QColor(rr, gg, bb, 45)))
                    p.drawPolygon(qpoly)
                    cx = sum(x for x, _ in pts)/len(pts)
                    cy = sum(y for _, y in pts)/len(pts)
                    p.setPen(QPen(col, 1)); p.setBrush(QBrush(col))
                    p.drawEllipse(int(cx*sw)-2, int(cy*sw)-2, 4, 4)
                    p.setPen(QPen(QColor(255, 255, 255)))
                    p.drawText(int(cx*sw)+5, int(cy*sw)-4, f["id"])
                    # 주차 전면(nose-in) 방향 화살표
                    if f["kind"] in ("vertical_parking", "parallel_parking"):
                        fy = -math.pi/2 if f["kind"] == "vertical_parking" else 0.0
                        ux, uy = -math.cos(fy), math.sin(fy)
                        L = max(14, int(20*sw))
                        ax, ay = int(cx*sw), int(cy*sw)
                        ex, ey = ax + int(L*ux), ay + int(L*uy)
                        ang = math.atan2(uy, ux)
                        p.setPen(QPen(col, 2)); p.drawLine(ax, ay, ex, ey)
                        for da in (math.pi - 0.4, math.pi + 0.4):
                            p.drawLine(ex, ey, ex + int(7*math.cos(ang+da)),
                                       ey + int(7*math.sin(ang+da)))
                    if edit:
                        for hi, (x, y) in enumerate(pts):
                            _handle(x, y, col, self._feat_hover == (fi, hi))
                elif f["type"] == "line":
                    (x0, y0), (x1, y1) = pts
                    p.setPen(QPen(col, 4))
                    p.drawLine(int(x0*sw), int(y0*sw), int(x1*sw), int(y1*sw))
                    p.setPen(QPen(QColor(255, 255, 255)))
                    p.drawText(int(x0*sw)+6, int(y0*sw), f["id"])
                    if edit:
                        for hi, (x, y) in enumerate(pts):
                            _handle(x, y, col, self._feat_hover == (fi, hi))
                elif f["type"] == "point":
                    x, y = pts[0]; sx, sy = int(x*sw), int(y*sw)
                    yaw = float(f.get("yaw", START_YAW))
                    if f["kind"] == "start_point":
                        # 스폰 origin ghost (점선) + 연결선
                        ox, oy, _yy = spawn_pose_from_dot(x, y, yaw)
                        gpx, gpy = world_to_pixel(ox, oy)
                        gsx, gsy = int(gpx*sw), int(gpy*sw)
                        p.setPen(QPen(QColor(rr, gg, bb, 140), 1, Qt.DashLine))
                        p.setBrush(Qt.NoBrush)
                        p.drawLine(sx, sy, gsx, gsy)
                        p.drawEllipse(gsx-3, gsy-3, 6, 6)
                    # heading 화살표 (픽셀방향 = (-cos yaw, sin yaw))
                    ux, uy = -math.cos(yaw), math.sin(yaw)
                    L = max(18, int(26*sw))
                    ex, ey = sx + int(L*ux), sy + int(L*uy)
                    ang = math.atan2(uy, ux)
                    p.setPen(QPen(col, 2)); p.drawLine(sx, sy, ex, ey)
                    for da in (math.pi - 0.4, math.pi + 0.4):
                        p.drawLine(ex, ey, ex + int(9*math.cos(ang+da)),
                                   ey + int(9*math.sin(ang+da)))
                    # 마커
                    hot = self._feat_hover == (fi, 0)
                    if f["kind"] == "start_point":
                        p.setPen(Qt.NoPen)
                        p.setBrush(QBrush(QColor(255, 255, 0) if hot else col))
                        r0 = 6 if hot else 5
                        p.drawEllipse(sx-r0, sy-r0, r0*2, r0*2)
                    else:   # obstacle / traffic_light: 속 빈 사각형 + X
                        r0 = 7 if hot else 6
                        p.setBrush(Qt.NoBrush)
                        p.setPen(QPen(QColor(255, 255, 0) if hot else col, 2))
                        p.drawRect(sx-r0, sy-r0, r0*2, r0*2)
                        p.drawLine(sx-r0, sy-r0, sx+r0, sy+r0)
                        p.drawLine(sx-r0, sy+r0, sx+r0, sy-r0)
                    p.setPen(QPen(QColor(255, 255, 255)))
                    p.drawText(sx+8, sy-6, f["id"])

        # 커서
        hx, hy = int(self._hover.x()*sw), int(self._hover.y()*sw)
        if self._mode == "mask_poly":
            # ── 폴리곤 미리보기 ───────────────────────────────────────────────
            col = QColor(255, 220, 60) if self._poly_fill_val == 255 else QColor(255, 80, 80)
            scr = [(int(x*sw), int(y*sw)) for x, y in self._poly_pts]

            if len(scr) >= 1:
                # 기존 꼭짓점 연결선
                p.setPen(QPen(col, 1.5, Qt.SolidLine)); p.setBrush(Qt.NoBrush)
                for i in range(1, len(scr)):
                    p.drawLine(scr[i-1][0], scr[i-1][1], scr[i][0], scr[i][1])
                # 마우스 → 마지막 꼭짓점 (점선)
                p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 140), 1.2, Qt.DashLine))
                p.drawLine(scr[-1][0], scr[-1][1], hx, hy)
                # 마우스 → 첫 꼭짓점 (닫히는 예고, >= 2점)
                if len(scr) >= 2:
                    p.drawLine(hx, hy, scr[0][0], scr[0][1])

            # 꼭짓점 점 (작게 - 차선 안 가리도록)
            p.setPen(Qt.NoPen); p.setBrush(QBrush(col))
            for sx, sy in scr:
                p.drawEllipse(sx-2, sy-2, 4, 4)
            # 첫 꼭짓점 (닫기 타겟): 조금 더 크게 + 녹색
            if scr:
                p.setBrush(QBrush(QColor(0, 255, 100, 230)))
                p.drawEllipse(scr[0][0]-4, scr[0][1]-4, 8, 8)

            # 현재 커서 십자
            p.setPen(QPen(col, 1.0))
            p.drawLine(hx-8, hy, hx+8, hy); p.drawLine(hx, hy-8, hx, hy+8)

        elif self._mode == "box_erase" and self._box_start is not None and self._drag:
            # 선택 사각형 미리보기
            bx, by = int(self._box_start.x()*sw), int(self._box_start.y()*sw)
            p.setPen(QPen(QColor(255, 80, 80, 220), 1.5, Qt.DashLine))
            p.setBrush(QBrush(QColor(255, 80, 80, 40)))
            rx, ry = min(bx, hx), min(by, hy)
            rw2, rh2 = abs(hx-bx), abs(hy-by)
            p.drawRect(rx, ry, rw2, rh2)
        else:
            p.setPen(QPen(QColor(255, 255, 0, 200), 1.0, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            br = max(4, int(self._brush*sw))
            p.drawEllipse(hx-br, hy-br, br*2, br*2)

        p.end()
        self.setPixmap(pm); self.resize(pm.size())

    def _overlay(self, p, mask, color, sw):
        mH, mW = mask.shape
        sW, sH = int(mW*sw), int(mH*sw)
        small = cv2.resize(mask, (sW, sH), interpolation=cv2.INTER_NEAREST)
        rgba  = np.zeros((sH, sW, 4), np.uint8)
        rgba[small > 128] = [color.red(), color.green(), color.blue(), color.alpha()]
        qi = QImage(rgba.data, sW, sH, sW*4, QImage.Format_RGBA8888)
        p.drawImage(0, 0, qi)

    @staticmethod
    def _to_pixmap(img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        return QPixmap.fromImage(QImage(rgb.data, w, h, w*c, QImage.Format_RGB888))

    def _s2i(self, sx, sy): return int(sx/self._scale), int(sy/self._scale)

    # ── 마우스 ────────────────────────────────────────────────────────────────
    def mousePressEvent(self, e):
        if self._src is None: return
        px, py = self._s2i(e.position().x(), e.position().y())
        self._drag = True
        if self._mode == "picker":
            if 0 <= py < self._src.shape[0] and 0 <= px < self._src.shape[1]:
                b, g, r = self._src[py, px]
                self.colorPicked.emit(px, py, int(b), int(g), int(r))
            return
        if self._mode == "box_erase":
            self._box_start = QPoint(px, py)
            return
        if self._mode == "poly_edit":
            if e.button() == Qt.LeftButton:
                idx = self._find_near_vertex(px, py)
                self._edit_drag_idx = idx
            elif e.button() == Qt.RightButton:
                self.delete_near_vertex(px, py)
            return
        if self._mode == "feature":
            if e.button() == Qt.LeftButton:
                if self._feat_add_start:
                    self.add_start_point(px, py, self._feat_add_side); return
                hit = self._feat_hit(px, py)
                if hit:
                    kind, fi, hi = hit
                    self._feat_drag = (fi, hi if kind == "handle" else "move")
                    self._feat_anchor = (px, py)
            elif e.button() == Qt.RightButton:
                self.delete_feature_at(px, py)
            return
        if self._mode == "mask_poly":
            if e.button() == Qt.LeftButton:
                # 첫 꼭짓점에 가까우면 닫기 (거리 < 20px)
                if (len(self._poly_pts) >= 3 and
                        math.hypot(px - self._poly_pts[0][0],
                                   py - self._poly_pts[0][1]) < 20 / self._scale):
                    self._close_polygon()
                else:
                    self._poly_pts.append((px, py))
                    self._refresh()
            elif e.button() == Qt.RightButton:
                self.cancel_polygon()
            return
        if self._mode == "draw":
            if e.button() == Qt.LeftButton: self._add(px, py)
            elif e.button() == Qt.RightButton: self._rem(px, py)
        elif self._mode == "erase": self._rem(px, py)

    def mouseMoveEvent(self, e):
        px, py = self._s2i(e.position().x(), e.position().y())
        self._hover = QPoint(px, py); self.hoverMoved.emit(px, py)
        if self._mode == "poly_edit":
            self._edit_hover_idx = self._find_near_vertex(px, py)
            if self._drag and self._edit_drag_idx >= 0:
                poly = self._get_active_edit_poly()
                if poly and 0 <= self._edit_drag_idx < len(poly):
                    poly[self._edit_drag_idx] = (px, py)
                    self._set_active_edit_poly(poly)
                    self._edit_poly_to_mask(self._edit_active)
            self.setCursor(
                Qt.SizeAllCursor if self._edit_hover_idx >= 0 else Qt.ArrowCursor)
            self._refresh(); return
        if self._mode == "feature":
            hit = self._feat_hit(px, py)
            self._feat_hover = (hit[1], hit[2]) if hit and hit[0] == "handle" else None
            if self._drag and self._feat_drag is not None:
                fi, hi = self._feat_drag
                f = self._feats[fi]
                if hi == "move":
                    ax, ay = self._feat_anchor
                    f["pts"] = [(x + (px - ax), y + (py - ay)) for x, y in f["pts"]]
                    self._feat_anchor = (px, py)
                else:
                    f["pts"][hi] = (px, py)
            self.setCursor(Qt.PointingHandCursor if self._feat_add_start else
                           (Qt.SizeAllCursor if hit else Qt.ArrowCursor))
            self._refresh(); return
        if self._drag and self._mode not in ("picker", "box_erase", "mask_poly"):
            if self._mode == "draw" and (e.buttons() & Qt.LeftButton):
                if not self._manual or self._dl(px, py) > max(2, self._brush//2):
                    self._add(px, py)
            elif self._mode == "erase": self._rem(px, py)
        self._refresh()

    def mouseReleaseEvent(self, e):
        if self._mode == "box_erase" and self._box_start is not None and self._drag:
            px, py = self._s2i(e.position().x(), e.position().y())
            x0, y0 = self._box_start.x(), self._box_start.y()
            x1, y1 = px, py
            xmin, xmax = min(x0, x1), max(x0, x1)
            ymin, ymax = min(y0, y1), max(y0, y1)
            before = len(self._manual)
            self._manual = [p for p in self._manual
                            if not (xmin <= p[0] <= xmax and ymin <= p[1] <= ymax)]
            removed = before - len(self._manual)
            self.pointAdded.emit()   # count 업데이트 재사용
            self._box_start = None
            self._refresh()
            if removed:
                from PySide6.QtCore import QTimer
                QTimer.singleShot(0, lambda: None)   # trigger parent update
        if self._mode == "poly_edit":
            self._edit_drag_idx = -1
        if self._mode == "feature":
            self._feat_drag = None
        self._drag = False

    def mouseDoubleClickEvent(self, e):
        """더블클릭: 폴리곤 닫기."""
        if self._mode == "mask_poly" and e.button() == Qt.LeftButton:
            self._close_polygon()

    def keyPressEvent(self, e):
        """Enter: 폴리곤 닫기   Escape: 취소   Delete/Back: 꼭짓점 삭제."""
        if self._mode == "mask_poly":
            if e.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._close_polygon()
            elif e.key() == Qt.Key_Escape:
                self.cancel_polygon()
        elif self._mode == "poly_edit":
            if e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
                hx, hy = self._hover.x(), self._hover.y()
                self.delete_near_vertex(hx, hy)
        super().keyPressEvent(e)

    def wheelEvent(self, e):
        self.set_scale(self._scale * (1.1 if e.angleDelta().y() > 0 else 1/1.1))

    def _add(self, px, py):
        h, w = self._src.shape[:2]
        self._manual.append((max(0, min(w-1, px)), max(0, min(h-1, py))))
        self.pointAdded.emit(); self._refresh()

    def _rem(self, px, py):
        n = len(self._manual)
        self._manual = [p for p in self._manual
                        if math.hypot(p[0]-px, p[1]-py) > self._brush]
        if len(self._manual) != n: self._refresh()

    def _dl(self, px, py):
        lx, ly = self._manual[-1]; return math.hypot(px-lx, py-ly)

    # ── 편집 상태 저장/불러오기 ───────────────────────────────────────────────
    def get_edit_state(self) -> dict:
        """현재 편집 상태(폴리곤 꼭짓점 + 수동 점)를 dict로 반환."""
        return {
            "version": 2,
            "edit_polygons": {
                "inner_out": self._edit_poly_i_out,
                "inner_inn": self._edit_poly_i_inn,
                "outer_out": self._edit_poly_o_out,
                "outer_inn": self._edit_poly_o_inn,
            },
            "manual_points": self._manual,
        }

    def set_edit_state(self, state: dict):
        """저장된 편집 상태 복원 후 마스크 재생성."""
        polys = state.get("edit_polygons", {})
        self._edit_poly_i_out = [tuple(p) for p in polys.get("inner_out", [])]
        self._edit_poly_i_inn = [tuple(p) for p in polys.get("inner_inn", [])]
        self._edit_poly_o_out = [tuple(p) for p in polys.get("outer_out", [])]
        self._edit_poly_o_inn = [tuple(p) for p in polys.get("outer_inn", [])]
        self._manual = [tuple(p) for p in state.get("manual_points", [])]
        # 폴리곤 → 마스크 재생성
        if self._edit_poly_i_out:
            self._edit_poly_to_mask("inner")
        if self._edit_poly_o_out:
            self._edit_poly_to_mask("outer")
        self._edit_drag_idx = -1
        self._edit_hover_idx = -1
        self._refresh()


# ─── 메인 윈도우 ──────────────────────────────────────────────────────────────
class GTAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Track GT Annotator v4")
        self.resize(1500, 920)

        self._gt_path    = DEFAULT_GT_PATH
        self._edit_state_path = os.path.expanduser("~/track_gt_edit_state.json")
        self._feat_path  = DEFAULT_FEATURES_PATH
        self._track_path = ""
        self._worker     = None
        self._seed_bgr   = None   # 샘플된 BGR

        self._build_ui()
        self._statusbar = QStatusBar(); self.setStatusBar(self._statusbar)
        self._set_status("준비.  Step 1 → 도로 위 클릭(색상 샘플)  Step 2 → 검출+분리  Step 3 → 저장")

        if os.path.exists(TRACK_PNG_PATH):
            self._load(TRACK_PNG_PATH)

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        cw = QWidget(); self.setCentralWidget(cw)
        root = QHBoxLayout(cw); root.setContentsMargins(4, 4, 4, 4); root.setSpacing(6)

        # ── 왼쪽: 스크롤 가능한 컨트롤 패널 ────────────────────────────────────
        scroll_ctrl = QScrollArea()
        scroll_ctrl.setWidgetResizable(True)
        scroll_ctrl.setFixedWidth(360)
        scroll_ctrl.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        ctrl = QWidget()
        cl = QVBoxLayout(ctrl); cl.setContentsMargins(6, 6, 6, 6); cl.setSpacing(8)

        # ── 수동 마스크 편집 (메인 워크플로우) ──────────────────────────────────
        g_mask = QGroupBox("🖌️  수동 마스크 편집  ← 주 워크플로우")
        g_mask.setStyleSheet(
            "QGroupBox{border:2px solid #4a8; color:#4fa; font-weight:bold;"
            " padding-top:8px; margin-top:4px;}"
            "QGroupBox::title{subcontrol-origin:margin; left:8px;}")
        lmask = QVBoxLayout(g_mask)

        # 자동 검출 마스크 가져오기
        btn_import = QPushButton("🔄  자동 검출 마스크 → 수동으로 가져오기")
        btn_import.setMinimumHeight(34)
        btn_import.setStyleSheet(
            "QPushButton{background:#3a3a1a; color:#ffd; font-weight:bold; border-radius:4px;}"
            "QPushButton:hover{background:#5a5a2a;}")
        btn_import.setToolTip(
            "Step 2 자동 검출 결과를 수동 마스크로 복사합니다.\n"
            "이후 폴리곤 채우기/지우기로 세밀하게 수정할 수 있습니다.")
        btn_import.clicked.connect(self._import_auto_masks)
        lmask.addWidget(btn_import)

        sep_m = QFrame(); sep_m.setFrameShape(QFrame.HLine)
        sep_m.setStyleSheet("color:#444"); lmask.addWidget(sep_m)

        # 현재 편집 차선 선택
        lmask.addWidget(QLabel("현재 편집 차선:"))
        self._lane_paint_bg = QButtonGroup(self)
        rb_paint_i = QRadioButton("🔴 1차선 (inner)"); rb_paint_i.setChecked(True)
        rb_paint_o = QRadioButton("🔵 2차선 (outer)")
        rb_paint_i.setStyleSheet("color:#ff8080; font-weight:bold;")
        rb_paint_o.setStyleSheet("color:#8080ff; font-weight:bold;")
        self._lane_paint_bg.addButton(rb_paint_i, 0)
        self._lane_paint_bg.addButton(rb_paint_o, 1)
        def _on_lane_changed(i):
            lane = ["inner", "outer"][i]
            self._canvas.set_active_lane(lane)   # 마스크 칠하기용
            self._canvas.set_edit_lane(lane)     # 꼭짓점 편집용 (_edit_active 동기화)
        self._lane_paint_bg.idClicked.connect(_on_lane_changed)
        lane_sel_row = QHBoxLayout()
        lane_sel_row.addWidget(rb_paint_i); lane_sel_row.addWidget(rb_paint_o)
        lmask.addLayout(lane_sel_row)

        # 폴리곤 모드 버튼 (채우기 / 지우기)
        poly_mode_row = QHBoxLayout()
        self._btn_poly_add = QPushButton("🔷 폴리곤 채우기")
        self._btn_poly_add.setCheckable(True)
        self._btn_poly_add.setMinimumHeight(36)
        self._btn_poly_add.setStyleSheet(
            "QPushButton{background:#1a4a2a; color:#afd; font-weight:bold; border-radius:4px;}"
            "QPushButton:checked{background:#2a8a4a; color:white; border:2px solid #4fa;}"
            "QPushButton:hover{background:#2a6a3a;}")
        self._btn_poly_erase = QPushButton("⬜ 폴리곤 지우기")
        self._btn_poly_erase.setCheckable(True)
        self._btn_poly_erase.setMinimumHeight(36)
        self._btn_poly_erase.setStyleSheet(
            "QPushButton{background:#3a1a1a; color:#faa; font-weight:bold; border-radius:4px;}"
            "QPushButton:checked{background:#8a2a2a; color:white; border:2px solid #f66;}"
            "QPushButton:hover{background:#5a2a2a;}")

        def _activate_poly(fill: bool):
            """폴리곤 채우기(fill=True) 또는 지우기(fill=False) 모드 활성화."""
            self._btn_poly_add.setChecked(fill)
            self._btn_poly_erase.setChecked(not fill)
            self._canvas.set_active_lane(
                ["inner","outer"][self._lane_paint_bg.checkedId()])
            self._canvas._poly_fill_val = 255 if fill else 0
            self._canvas.set_mode("mask_poly")
            self._canvas.setFocus()
            # 기존 모드 라디오 해제
            self._mode_bg.setExclusive(False)
            for rb in self._mode_bg.buttons(): rb.setChecked(False)
            self._mode_bg.setExclusive(True)

        def _deactivate_poly():
            """폴리곤 모드 해제 → 드로우 모드 복귀."""
            self._btn_poly_add.setChecked(False)
            self._btn_poly_erase.setChecked(False)
            self._canvas.cancel_polygon()
            self._canvas.set_mode("draw")
            self._mode_bg.buttons()[0].setChecked(True)

        self._btn_poly_add.clicked.connect(
            lambda c: _activate_poly(True) if c else _deactivate_poly())
        self._btn_poly_erase.clicked.connect(
            lambda c: _activate_poly(False) if c else _deactivate_poly())

        poly_mode_row.addWidget(self._btn_poly_add)
        poly_mode_row.addWidget(self._btn_poly_erase)
        lmask.addLayout(poly_mode_row)

        poly_tip = QLabel(
            "  클릭: 꼭짓점 추가\n"
            "  더블클릭 / Enter: 폴리곤 닫기(채움)\n"
            "  첫 꼭짓점 근처 클릭: 자동 닫기\n"
            "  우클릭 / Esc: 취소"
        )
        poly_tip.setStyleSheet("color:#8cf; font-size:9px;")
        lmask.addWidget(poly_tip)

        # Undo 버튼
        undo_row = QHBoxLayout()
        btn_poly_undo = QPushButton("↩ 폴리곤 되돌리기")
        btn_poly_undo.clicked.connect(self._undo_polygon)
        btn_poly_cancel = QPushButton("✖ 현재 취소")
        btn_poly_cancel.clicked.connect(self._cancel_polygon)
        undo_row.addWidget(btn_poly_undo); undo_row.addWidget(btn_poly_cancel)
        lmask.addLayout(undo_row)

        # 중앙선 추출
        btn_extract = QPushButton("📐  마스크에서 중앙선 추출")
        btn_extract.setMinimumHeight(38)
        btn_extract.setStyleSheet(
            "QPushButton{background:#1a4a7a; color:white; font-weight:bold;"
            " font-size:12px; border-radius:5px;}"
            "QPushButton:hover{background:#2a6aaa;}")
        btn_extract.clicked.connect(self._extract_from_masks)
        lmask.addWidget(btn_extract)

        sep_m2 = QFrame(); sep_m2.setFrameShape(QFrame.HLine)
        sep_m2.setStyleSheet("color:#444"); lmask.addWidget(sep_m2)

        # ── 편집 폴리곤 (꼭짓점 드래그) ───────────────────────────────────────
        lmask.addWidget(QLabel("✏️  꼭짓점 드래그 편집:"))

        # 편집할 경계 선택 (외곽 / 내곽 구멍)
        lmask.addWidget(QLabel("편집할 경계:"))
        self._boundary_bg = QButtonGroup(self)
        rb_bnd_out = QRadioButton("▣ 외곽 경계"); rb_bnd_out.setChecked(True)
        rb_bnd_inn = QRadioButton("◻ 내곽 구멍 (링)")
        rb_bnd_out.setToolTip("바깥 경계선을 편집합니다 (실선으로 표시)")
        rb_bnd_inn.setToolTip("내부 구멍 경계를 편집합니다 (점선으로 표시)\n"
                              "링 형태 마스크에서 안쪽 경계를 조정할 때 사용")
        self._boundary_bg.addButton(rb_bnd_out, 0)
        self._boundary_bg.addButton(rb_bnd_inn, 1)
        self._boundary_bg.idClicked.connect(
            lambda i: self._canvas.set_edit_boundary(["outer","inner"][i]))
        bnd_row = QHBoxLayout()
        bnd_row.addWidget(rb_bnd_out); bnd_row.addWidget(rb_bnd_inn)
        lmask.addLayout(bnd_row)

        # 꼭짓점 수 선택
        npts_row = QHBoxLayout(); npts_row.addWidget(QLabel("꼭짓점 수:"))
        self._npts_sp = QSpinBox()
        self._npts_sp.setRange(20, 300); self._npts_sp.setValue(80)
        self._npts_sp.setToolTip("마스크 → 폴리곤 변환 시 꼭짓점 개수 (적을수록 편집 쉬움)")
        npts_row.addWidget(self._npts_sp)
        lmask.addLayout(npts_row)

        # 마스크 → 편집 폴리곤 변환 버튼
        btn_to_poly = QPushButton("📐  마스크 → 편집 폴리곤 변환")
        btn_to_poly.setMinimumHeight(34)
        btn_to_poly.setStyleSheet(
            "QPushButton{background:#2a2a4a; color:#aaf; font-weight:bold; border-radius:4px;}"
            "QPushButton:hover{background:#3a3a7a;}")
        btn_to_poly.setToolTip(
            "현재 수동 마스크의 외곽선을 편집 폴리곤으로 변환합니다.\n"
            "(자동 마스크를 가져온 후 사용)")
        btn_to_poly.clicked.connect(self._mask_to_edit_poly)
        lmask.addWidget(btn_to_poly)

        # 꼭짓점 편집 모드 토글
        self._btn_poly_edit = QPushButton("🖱️  꼭짓점 편집 모드 OFF")
        self._btn_poly_edit.setCheckable(True)
        self._btn_poly_edit.setMinimumHeight(34)
        self._btn_poly_edit.setStyleSheet(
            "QPushButton{background:#2a2a2a; color:#ccc; font-weight:bold; border-radius:4px;}"
            "QPushButton:checked{background:#4a3a7a; color:white; border:2px solid #88f;}"
            "QPushButton:hover{background:#3a3a4a;}")

        def _toggle_poly_edit(checked):
            if checked:
                self._btn_poly_edit.setText("🖱️  꼭짓점 편집 모드 ON")
                self._canvas.set_edit_lane(
                    ["inner","outer"][self._lane_paint_bg.checkedId()])
                self._canvas.set_mode("poly_edit")
                self._canvas.setFocus()
                # 다른 모드 버튼 해제
                self._btn_poly_add.setChecked(False)
                self._btn_poly_erase.setChecked(False)
                self._mode_bg.setExclusive(False)
                for rb in self._mode_bg.buttons(): rb.setChecked(False)
                self._mode_bg.setExclusive(True)
                self._set_status(
                    "꼭짓점 편집 모드: 꼭짓점 드래그=이동  우클릭=삭제  "
                    "편집 차선은 위에서 선택")
            else:
                self._btn_poly_edit.setText("🖱️  꼭짓점 편집 모드 OFF")
                self._canvas.set_mode("draw")
                self._mode_bg.buttons()[0].setChecked(True)

        self._btn_poly_edit.toggled.connect(_toggle_poly_edit)
        lmask.addWidget(self._btn_poly_edit)

        edit_tip = QLabel(
            "  드래그: 꼭짓점 이동 (마스크 즉시 반영)\n"
            "  우클릭: 꼭짓점 삭제\n"
            "  편집 후 '마스크에서 중앙선 추출'"
        )
        edit_tip.setStyleSheet("color:#8cf; font-size:9px;")
        lmask.addWidget(edit_tip)

        sep_m3 = QFrame(); sep_m3.setFrameShape(QFrame.HLine)
        sep_m3.setStyleSheet("color:#444"); lmask.addWidget(sep_m3)

        # 마스크 초기화
        mask_clr_row = QHBoxLayout()
        btn_clr_i = QPushButton("🔴 1차선 초기화")
        btn_clr_o = QPushButton("🔵 2차선 초기화")
        btn_clr_i.setStyleSheet("color:#ff8080;")
        btn_clr_o.setStyleSheet("color:#8080ff;")
        btn_clr_i.clicked.connect(lambda: self._canvas.clear_mask("inner"))
        btn_clr_o.clicked.connect(lambda: self._canvas.clear_mask("outer"))
        mask_clr_row.addWidget(btn_clr_i); mask_clr_row.addWidget(btn_clr_o)
        lmask.addLayout(mask_clr_row)

        # 마스크 결과 표시
        self._lbl_mask_result = QLabel("마스크 중앙선: 추출 전")
        self._lbl_mask_result.setFont(QFont("Monospace", 9))
        self._lbl_mask_result.setStyleSheet(
            "color:#8f8; padding:3px; background:#1a2a1a; border-radius:3px;")
        self._lbl_mask_result.setWordWrap(True)
        lmask.addWidget(self._lbl_mask_result)

        cl.addWidget(g_mask)

        # ── 🅵 피처 주석 (주차칸 / IN·OUT선 / 횡단보도 / 출발점) ──────────────────
        g_feat = QGroupBox("🅵  피처 주석  (주차칸·IN/OUT·횡단보도·출발점)")
        g_feat.setStyleSheet(
            "QGroupBox{border:2px solid #a84; color:#fc8; font-weight:bold;"
            " padding-top:8px; margin-top:4px;}"
            "QGroupBox::title{subcontrol-origin:margin; left:8px;}")
        lfeat = QVBoxLayout(g_feat)

        btn_feat_auto = QPushButton("🔄  자동 검출 피처 로드")
        btn_feat_auto.setMinimumHeight(34)
        btn_feat_auto.setStyleSheet(
            "QPushButton{background:#4a3a1a; color:#fd8; font-weight:bold; border-radius:4px;}"
            "QPushButton:hover{background:#6a5a2a;}")
        btn_feat_auto.setToolTip("track.png 측정값 기반 기본 피처를 캔버스에 올립니다.")
        btn_feat_auto.clicked.connect(self._load_auto_features)
        lfeat.addWidget(btn_feat_auto)

        self._btn_feat_edit = QPushButton("🖱️  피처 편집 모드 OFF")
        self._btn_feat_edit.setCheckable(True)
        self._btn_feat_edit.setMinimumHeight(34)
        self._btn_feat_edit.setStyleSheet(
            "QPushButton{background:#2a2a2a; color:#ccc; font-weight:bold; border-radius:4px;}"
            "QPushButton:checked{background:#7a5a1a; color:white; border:2px solid #fc6;}"
            "QPushButton:hover{background:#3a3a3a;}")
        self._btn_feat_edit.toggled.connect(self._toggle_feature_edit)
        lfeat.addWidget(self._btn_feat_edit)

        self._btn_feat_addstart = QPushButton("➕  출발점 추가 (클릭) OFF")
        self._btn_feat_addstart.setCheckable(True)
        self._btn_feat_addstart.setMinimumHeight(30)
        self._btn_feat_addstart.setStyleSheet(
            "QPushButton{background:#3a1a1a; color:#faa; font-weight:bold; border-radius:4px;}"
            "QPushButton:checked{background:#8a2a2a; color:white; border:2px solid #f66;}")
        self._btn_feat_addstart.toggled.connect(self._toggle_add_start)
        lfeat.addWidget(self._btn_feat_addstart)

        side_row = QHBoxLayout()
        side_row.addWidget(QLabel("추가할 측:"))
        self._feat_side_combo = QComboBox()
        self._feat_side_combo.addItems(["IN (서쪽 yaw=0)", "OUT (동쪽 yaw=π)"])
        self._feat_side_combo.currentIndexChanged.connect(
            lambda i: self._canvas.set_feat_add_side(["IN", "OUT"][i]))
        side_row.addWidget(self._feat_side_combo)
        lfeat.addLayout(side_row)

        self._chk_feat = QCheckBox("피처 표시"); self._chk_feat.setChecked(True)
        self._chk_feat.toggled.connect(lambda v: self._canvas.toggle_features(v))
        lfeat.addWidget(self._chk_feat)

        feat_tip = QLabel(
            "  편집 모드: 코너/끝점 드래그=수정, 박스/선 내부 드래그=이동\n"
            "  출발점: 우클릭=삭제, '출발점 추가' ON 후 클릭=생성\n"
            "  점선=오른쪽 앞바퀴 정렬 시 base_link 스폰 위치"
        )
        feat_tip.setStyleSheet("color:#fc8; font-size:9px;")
        feat_tip.setWordWrap(True)
        lfeat.addWidget(feat_tip)

        self._lbl_feat_path = QLabel(f"경로:\n{self._feat_path}")
        self._lbl_feat_path.setWordWrap(True)
        self._lbl_feat_path.setStyleSheet("color:#aaa; font-size:9px")
        lfeat.addWidget(self._lbl_feat_path)

        feat_io = QHBoxLayout()
        btn_feat_save = QPushButton("💾 피처 저장")
        btn_feat_load = QPushButton("📂 불러오기")
        btn_feat_save.setMinimumHeight(32)
        btn_feat_load.setMinimumHeight(32)
        btn_feat_save.setStyleSheet(
            "QPushButton{background:#4a3a1a; color:#fd8; font-weight:bold; border-radius:4px;}"
            "QPushButton:hover{background:#6a5a2a;}")
        btn_feat_load.setStyleSheet(
            "QPushButton{background:#1a3a5a; color:#adf; font-weight:bold; border-radius:4px;}"
            "QPushButton:hover{background:#2a5a8a;}")
        btn_feat_save.clicked.connect(self._save_features)
        btn_feat_load.clicked.connect(self._load_features_json)
        feat_io.addWidget(btn_feat_save); feat_io.addWidget(btn_feat_load)
        lfeat.addLayout(feat_io)

        btn_feat_path = QPushButton("경로 변경")
        btn_feat_path.clicked.connect(self._change_feat_path)
        lfeat.addWidget(btn_feat_path)

        self._lbl_feat_result = QLabel("피처: 없음")
        self._lbl_feat_result.setFont(QFont("Monospace", 9))
        self._lbl_feat_result.setStyleSheet(
            "color:#fc8; padding:3px; background:#2a2010; border-radius:3px;")
        self._lbl_feat_result.setWordWrap(True)
        lfeat.addWidget(self._lbl_feat_result)

        cl.addWidget(g_feat)

        # ── STEP 1: 도로 색상 샘플 ─────────────────────────────────────────────
        g1 = QGroupBox("Step 1  ·  도로 색상 샘플 (선택)"); l1 = QVBoxLayout(g1)

        # 색상 스워치
        swatch_row = QHBoxLayout()
        self._swatch = QLabel()
        self._swatch.setFixedSize(40, 40)
        self._swatch.setAlignment(Qt.AlignCenter)
        self._swatch.setStyleSheet("background:#444; border:1px solid #777; border-radius:4px;")
        self._lbl_bgr = QLabel("클릭 전")
        self._lbl_bgr.setFont(QFont("Monospace", 8))
        self._lbl_bgr.setWordWrap(True)
        swatch_row.addWidget(self._swatch)
        swatch_row.addWidget(self._lbl_bgr, stretch=1)
        l1.addLayout(swatch_row)

        self._mode_bg = QButtonGroup(self)
        rb_draw   = QRadioButton("✏️ 드로우"); rb_draw.setChecked(True)
        rb_erase  = QRadioButton("🧹 점 지우기")
        rb_picker = QRadioButton("🎨 색상 피커")
        rb_box    = QRadioButton("☐ 범위 지우기")
        rb_box.setStyleSheet("color:#ff9060; font-weight:bold;")
        for i, rb in enumerate([rb_draw, rb_erase, rb_picker, rb_box]):
            self._mode_bg.addButton(rb, i)
        self._mode_bg.idClicked.connect(lambda i: self._canvas.set_mode(
            ["draw", "erase", "picker", "box_erase"][i]))

        # 2행 배치
        mode_row1 = QHBoxLayout()
        mode_row2 = QHBoxLayout()
        mode_row1.addWidget(rb_draw);   mode_row1.addWidget(rb_erase)
        mode_row2.addWidget(rb_picker); mode_row2.addWidget(rb_box)
        l1.addLayout(mode_row1)
        l1.addLayout(mode_row2)

        tip = QLabel(
            "  💡 색상 피커: 회색 도로 클릭 → 색상 샘플\n"
            "  ☐ 범위 지우기: 드래그 → 사각형 안 점 일괄 삭제"
        )
        tip.setStyleSheet("color:#8cf; font-size:10px"); l1.addWidget(tip)
        cl.addWidget(g1)

        # ── STEP 2: 검출 + 차선 분리 ───────────────────────────────────────────
        g2 = QGroupBox("Step 2  ·  도로 검출 + 차선 분리"); l2 = QVBoxLayout(g2)

        # 허용 오차
        tol_row = QHBoxLayout(); tol_row.addWidget(QLabel("색상 허용 오차:"))
        self._tol_sl = QSlider(Qt.Horizontal); self._tol_sl.setRange(5, 80); self._tol_sl.setValue(25)
        self._lbl_tol = QLabel("25")
        self._tol_sl.valueChanged.connect(lambda v: self._lbl_tol.setText(str(v)))
        tol_row.addWidget(self._tol_sl); tol_row.addWidget(self._lbl_tol)
        l2.addLayout(tol_row)

        # 메인 실행 버튼 (색상 피커 기반)
        btn_run_color = QPushButton("🚀  샘플 색으로 검출 + 차선 분리")
        btn_run_color.setMinimumHeight(40)
        btn_run_color.setStyleSheet(
            "QPushButton{background:#1a6a2a; color:white; font-weight:bold; "
            "font-size:13px; border-radius:5px;}"
            "QPushButton:hover{background:#1e8a3a;}"
        )
        btn_run_color.clicked.connect(self._run_color)
        l2.addWidget(btn_run_color)

        # 구분선
        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#555"); l2.addWidget(sep)

        # V 범위 대안
        l2.addWidget(QLabel("또는  V 범위로 검출:"))
        self._sl_vmin = self._slider_row(l2, "V 최소:", 30, 200, 100)
        self._sl_vmax = self._slider_row(l2, "V 최대:", 50, 255, 150)
        hint2 = QLabel("  도로≈121  배경≈76  흰선≈231")
        hint2.setStyleSheet("color:#8af; font-size:9px"); l2.addWidget(hint2)

        btn_run_v = QPushButton("🔎  V 범위로 검출 + 차선 분리")
        btn_run_v.setMinimumHeight(34)
        btn_run_v.setStyleSheet(
            "QPushButton{background:#1a4a6a; color:white; font-weight:bold; border-radius:4px;}"
            "QPushButton:hover{background:#1a5a8a;}"
        )
        btn_run_v.clicked.connect(self._run_v)
        l2.addWidget(btn_run_v)

        # 결과 정보
        self._lbl_result = QLabel("결과: 대기 중")
        self._lbl_result.setFont(QFont("Monospace", 9))
        self._lbl_result.setStyleSheet("color:#8f8; padding:4px; background:#1a2a1a; border-radius:3px;")
        self._lbl_result.setWordWrap(True)
        l2.addWidget(self._lbl_result)
        cl.addWidget(g2)

        # ── 차선 표시 선택 ──────────────────────────────────────────────────────
        g_view = QGroupBox("차선 표시"); lv = QVBoxLayout(g_view)
        self._lane_combo = QComboBox()
        self._lane_combo.addItems(["🔴🔵 둘 다 표시", "🔴 1차선(inner)만", "🔵 2차선(outer)만"])
        self._lane_combo.currentIndexChanged.connect(
            lambda i: self._canvas.set_show_lane(["both", "inner", "outer"][i]))
        lv.addWidget(self._lane_combo)

        # 차선 복사 버튼 (2개 나란히)
        cp_row = QHBoxLayout()
        btn_cp_i = QPushButton("⬇ 1차선 → 수동 복사")
        btn_cp_o = QPushButton("⬇ 2차선 → 수동 복사")
        btn_cp_i.setToolTip("1차선(inner) 중앙선을 수동 점에 복사")
        btn_cp_o.setToolTip("2차선(outer) 중앙선을 수동 점에 복사")
        btn_cp_i.clicked.connect(lambda: self._copy_lane("inner"))
        btn_cp_o.clicked.connect(lambda: self._copy_lane("outer"))
        cp_row.addWidget(btn_cp_i); cp_row.addWidget(btn_cp_o)
        lv.addLayout(cp_row)
        cl.addWidget(g_view)

        # ── 뷰 옵션 ────────────────────────────────────────────────────────────
        g_opt = QGroupBox("뷰 옵션"); lo = QVBoxLayout(g_opt)
        zoom_row = QHBoxLayout(); zoom_row.addWidget(QLabel("줌:"))
        self._zoom_sl = QSlider(Qt.Horizontal); self._zoom_sl.setRange(10, 300); self._zoom_sl.setValue(75)
        self._lbl_zoom = QLabel("75%")
        self._zoom_sl.valueChanged.connect(lambda v: (
            self._canvas.set_scale(v / 100.0), self._lbl_zoom.setText(f"{v}%")))
        zoom_row.addWidget(self._zoom_sl); zoom_row.addWidget(self._lbl_zoom)
        lo.addLayout(zoom_row)

        brush_row = QHBoxLayout(); brush_row.addWidget(QLabel("브러시:"))
        self._brush_sp = QSpinBox(); self._brush_sp.setRange(1, 100); self._brush_sp.setValue(10)
        self._brush_sp.valueChanged.connect(lambda v: self._canvas.set_brush(v))
        brush_row.addWidget(self._brush_sp)
        lo.addLayout(brush_row)

        chk_row = QHBoxLayout()
        self._chk_lm   = QCheckBox("랜드마크"); self._chk_lm.setChecked(True)
        self._chk_mask = QCheckBox("마스크");   self._chk_mask.setChecked(True)
        self._chk_lm.toggled.connect(lambda v: self._canvas.toggle_landmarks(v))
        self._chk_mask.toggled.connect(lambda v: self._canvas.toggle_mask(v))
        chk_row.addWidget(self._chk_lm); chk_row.addWidget(self._chk_mask)
        lo.addLayout(chk_row)
        cl.addWidget(g_opt)

        # ── 수동 드로우 관리 ───────────────────────────────────────────────────
        g_manual = QGroupBox("수동 드로우 관리"); lm2 = QVBoxLayout(g_manual)
        self._lbl_cnt = QLabel("수동 점: 0개")
        self._lbl_cnt.setFont(QFont("Monospace", 9)); lm2.addWidget(self._lbl_cnt)
        row_m = QHBoxLayout()
        btn_del = QPushButton("전체 삭제"); btn_del.clicked.connect(self._clear_manual)
        btn_sort = QPushButton("🔀 순서 정렬"); btn_sort.clicked.connect(self._sort)
        row_m.addWidget(btn_del); row_m.addWidget(btn_sort); lm2.addLayout(row_m)
        cl.addWidget(g_manual)

        # ── STEP 3: 저장 ────────────────────────────────────────────────────────
        g3 = QGroupBox("Step 3  ·  저장"); l3 = QVBoxLayout(g3)
        l3.addWidget(QLabel("내보낼 차선:"))
        self._export_combo = QComboBox()
        self._export_combo.addItems([
            "🔴 1차선 (inner) — 마스크 추출 결과",
            "🔵 2차선 (outer) — 마스크 추출 결과",
            "✏️ 수동 점만 (draw 모드 점)",
        ])
        l3.addWidget(self._export_combo)

        self._lbl_gt = QLabel(f"경로:\n{DEFAULT_GT_PATH}")
        self._lbl_gt.setWordWrap(True)
        self._lbl_gt.setStyleSheet("color:#aaa; font-size:9px"); l3.addWidget(self._lbl_gt)

        btn_path = QPushButton("경로 변경"); btn_path.clicked.connect(self._change_path)
        l3.addWidget(btn_path)

        btn_save = QPushButton("✅  GT JSON 저장")
        btn_save.setMinimumHeight(36)
        btn_save.setStyleSheet("background:#277; color:white; font-weight:bold; border-radius:4px;")
        btn_save.clicked.connect(self._save_gt); l3.addWidget(btn_save)

        btn_py = QPushButton("🐍  track_data.py 업데이트")
        btn_py.clicked.connect(self._export_py); l3.addWidget(btn_py)
        cl.addWidget(g3)

        # ── 편집 상태 저장/불러오기 ────────────────────────────────────────────
        g_es = QGroupBox("💾  편집 상태 저장/불러오기"); les = QVBoxLayout(g_es)

        self._lbl_es_path = QLabel(
            f"경로:\n{os.path.expanduser('~/track_gt_edit_state.json')}")
        self._lbl_es_path.setWordWrap(True)
        self._lbl_es_path.setStyleSheet("color:#aaa; font-size:9px")
        les.addWidget(self._lbl_es_path)

        es_row = QHBoxLayout()
        btn_es_save = QPushButton("💾 저장")
        btn_es_load = QPushButton("📂 불러오기")
        btn_es_save.setMinimumHeight(32)
        btn_es_load.setMinimumHeight(32)
        btn_es_save.setStyleSheet(
            "QPushButton{background:#1a4a3a; color:#afa; font-weight:bold; border-radius:4px;}"
            "QPushButton:hover{background:#2a7a5a;}")
        btn_es_load.setStyleSheet(
            "QPushButton{background:#1a3a5a; color:#adf; font-weight:bold; border-radius:4px;}"
            "QPushButton:hover{background:#2a5a8a;}")
        btn_es_save.clicked.connect(self._save_edit_state)
        btn_es_load.clicked.connect(self._load_edit_state)
        es_row.addWidget(btn_es_save); es_row.addWidget(btn_es_load)
        les.addLayout(es_row)

        btn_es_path = QPushButton("경로 변경")
        btn_es_path.clicked.connect(self._change_es_path)
        les.addWidget(btn_es_path)

        hint_es = QLabel(
            "  폴리곤 꼭짓점 + 수동 점 저장\n"
            "  불러오면 마스크도 자동으로 복원됩니다"
        )
        hint_es.setStyleSheet("color:#8cf; font-size:9px")
        les.addWidget(hint_es)
        cl.addWidget(g_es)

        # ── 파일 열기 ──────────────────────────────────────────────────────────
        g_file = QGroupBox("파일"); lf = QVBoxLayout(g_file)
        btn_open = QPushButton("📂  트랙 이미지 열기"); btn_open.clicked.connect(self._open_dialog)
        lf.addWidget(btn_open)
        self._lbl_file = QLabel("자동 로드 중..."); self._lbl_file.setWordWrap(True)
        self._lbl_file.setStyleSheet("color:#aaa; font-size:9px"); lf.addWidget(self._lbl_file)
        cl.addWidget(g_file)

        cl.addStretch()

        # 커서 좌표
        g_coord = QGroupBox("📍 커서"); lc = QVBoxLayout(g_coord)
        self._lbl_cursor = QLabel("이미지: (-,-)\n세계: (-,-)  [m]")
        self._lbl_cursor.setFont(QFont("Monospace", 9)); lc.addWidget(self._lbl_cursor)
        cl.addWidget(g_coord)

        scroll_ctrl.setWidget(ctrl)

        # ── 오른쪽: 캔버스 ─────────────────────────────────────────────────────
        canvas_scroll = QScrollArea(); canvas_scroll.setWidgetResizable(False)
        self._canvas = Canvas()
        self._canvas.set_scale(0.75)
        self._canvas.hoverMoved.connect(self._on_hover)
        self._canvas.colorPicked.connect(self._on_picked)
        self._canvas.pointAdded.connect(self._update_count)
        canvas_scroll.setWidget(self._canvas)

        root.addWidget(scroll_ctrl)
        root.addWidget(canvas_scroll, stretch=1)

        # 툴바
        tb = self.addToolBar("툴바"); tb.setMovable(False)
        for lbl, sh, fn in [
            ("📂 열기",  QKeySequence.Open, self._open_dialog),
            ("💾 저장",  QKeySequence.Save, self._save_gt),
            ("↩ 실행취소", QKeySequence.Undo, self._undo),
        ]:
            a = QAction(lbl, self); a.setShortcut(sh); a.triggered.connect(fn); tb.addAction(a)

    @staticmethod
    def _slider_row(layout, label, lo, hi, default):
        row = QHBoxLayout(); row.addWidget(QLabel(label))
        sl = QSlider(Qt.Horizontal); sl.setRange(lo, hi); sl.setValue(default)
        lb = QLabel(str(default)); sl.valueChanged.connect(lambda v, l=lb: l.setText(str(v)))
        row.addWidget(sl); row.addWidget(lb); layout.addLayout(row)
        return sl

    # ── 슬롯 ─────────────────────────────────────────────────────────────────
    def _set_status(self, msg): self._statusbar.showMessage(msg)

    def _on_hover(self, px, py):
        wx, wy = pixel_to_world(px, py)
        self._lbl_cursor.setText(f"이미지: ({px},{py})\n세계: ({wx:.3f},{wy:.3f})  [m]")

    def _on_picked(self, px, py, b, g, r):
        self._seed_bgr = np.array([b, g, r], np.uint8)
        hx = f"#{r:02x}{g:02x}{b:02x}"
        brightness = 0.299*r + 0.587*g + 0.114*b
        fg = "#000" if brightness > 128 else "#fff"
        self._swatch.setStyleSheet(
            f"background:{hx}; color:{fg}; border:1px solid #999; border-radius:4px;")
        self._swatch.setText(hx)
        self._lbl_bgr.setText(f"BGR:({b},{g},{r})\nV={max(b,g,r)}")

        v = max(b, g, r)
        # 어두운 색 (배경) 경고
        if v < 90:
            self._set_status(f"⚠️  V={v} — 배경을 클릭했을 가능성이 높습니다! "
                             f"회색 도로(V≈120) 위를 클릭하세요.")
        # 흰색 (차선 마킹) 경고
        elif v > 200:
            self._set_status(f"⚠️  V={v} — 흰 차선 마킹을 클릭했습니다! "
                             f"회색 도로 표면(V≈120)을 클릭하거나 "
                             f"'V 범위로 검출' 버튼을 사용하세요.")
        else:
            self._set_status(f"색상 샘플: ({px},{py}) BGR=({b},{g},{r}) V={v}"
                             f"  →  '샘플 색으로 검출+분리' 버튼 클릭!")

    def _update_count(self):
        self._lbl_cnt.setText(f"수동 점: {len(self._canvas.get_manual())}개")

    def _undo(self):
        m = self._canvas.get_manual()
        if m: self._canvas.set_manual(m[:-1]); self._update_count()

    def _clear_manual(self):
        if QMessageBox.question(self, "확인", "수동 점을 전부 삭제할까요?") == QMessageBox.Yes:
            self._canvas.clear_manual(); self._update_count()

    def _sort(self):
        pts = self._canvas.get_manual()
        if len(pts) < 2: self._set_status("정렬할 점이 없습니다."); return
        arr = np.array(pts, float)
        used, sorted_ = set(), []
        cur = int(np.argmax(arr[:, 1]))
        while len(used) < len(arr):
            used.add(cur); sorted_.append(arr[cur])
            rem = [i for i in range(len(arr)) if i not in used]
            if not rem: break
            ds = [math.hypot(arr[i][0]-arr[cur][0], arr[i][1]-arr[cur][1]) for i in rem]
            ni = int(np.argmin(ds))
            if ds[ni] > 200: break
            cur = rem[ni]
        self._canvas.set_manual([(int(x), int(y)) for x, y in sorted_])
        self._update_count()
        self._set_status(f"정렬 완료: {len(sorted_)}개")

    def _mask_to_edit_poly(self):
        """현재 수동 마스크(양쪽) → 편집 폴리곤으로 변환."""
        n_out = self._npts_sp.value()
        n_inn = max(20, n_out * 3 // 4)   # 내곽은 약간 적게
        ok_i = self._canvas.import_mask_as_polygon("inner", n_out, n_inn)
        ok_o = self._canvas.import_mask_as_polygon("outer", n_out, n_inn)
        if ok_i or ok_o:
            msgs = []
            if ok_i:
                ni_o = len(self._canvas._edit_poly_i_out)
                ni_i = len(self._canvas._edit_poly_i_inn)
                hole_i = f"{ni_i}꼭짓점" if ni_i else "❌미검출"
                msgs.append(f"1차선 외곽:{ni_o}  내곽:{hole_i}")
            if ok_o:
                no_o = len(self._canvas._edit_poly_o_out)
                no_i = len(self._canvas._edit_poly_o_inn)
                hole_o = f"{no_i}꼭짓점" if no_i else "❌미검출"
                msgs.append(f"2차선 외곽:{no_o}  내곽:{hole_o}")
            self._set_status(
                f"편집 폴리곤 변환 완료: {', '.join(msgs)}  "
                "— '꼭짓점 편집 모드' ON 후 드래그로 수정하세요.")
            self._lbl_mask_result.setText(
                "✅ 편집 폴리곤 준비\n"
                + "\n".join(msgs)
                + "\n실선=외곽경계  점선=내곽구멍"
            )
        else:
            self._set_status(
                "⚠️  변환할 마스크가 없습니다. "
                "먼저 자동 검출 후 '가져오기' 또는 폴리곤 채우기를 실행하세요.")

    def _import_auto_masks(self):
        ok = self._canvas.import_auto_masks()
        if ok:
            ni = int((self._canvas._mask_i > 0).sum())
            no = int((self._canvas._mask_o > 0).sum())
            self._lbl_mask_result.setText(
                f"✅ 자동 마스크 가져옴\n"
                f"   🔴 1차선: {ni:,}px\n"
                f"   🔵 2차선: {no:,}px\n"
                f"   폴리곤으로 수정 후 '중앙선 추출'"
            )
            self._set_status(
                "자동 마스크 복사 완료! 폴리곤 채우기/지우기로 경계를 수정하세요.")
        else:
            self._set_status("⚠️  먼저 Step 2에서 자동 검출을 실행하세요.")

    def _undo_polygon(self):
        ok = self._canvas.undo_polygon()
        self._set_status("폴리곤 되돌리기 완료" if ok else "되돌릴 히스토리 없음")

    def _cancel_polygon(self):
        self._canvas.cancel_polygon()
        self._set_status("폴리곤 취소")

    def _extract_from_masks(self):
        """수동 마스크에서 거리변환 중앙선 추출."""
        if self._canvas._mask_i is None or self._canvas._mask_o is None:
            self._set_status("이미지를 먼저 로드하세요."); return
        ni_px = int((self._canvas._mask_i > 0).sum())
        no_px = int((self._canvas._mask_o > 0).sum())
        if ni_px == 0 and no_px == 0:
            self._set_status("⚠️ 마스크가 비어 있습니다. 먼저 칠해주세요."); return
        self._set_status("중앙선 추출 중...")
        ni, no = self._canvas.extract_from_masks()

        # ── 내보낼 차선 콤보 자동 설정 ─────────────────────────────────────
        # 추출된 점이 있는 차선을 자동 선택 (수동점만 → 차선으로 전환)
        if ni > 0 and no > 0:
            self._export_combo.setCurrentIndex(0)   # 1차선 기본
        elif ni > 0:
            self._export_combo.setCurrentIndex(0)
        elif no > 0:
            self._export_combo.setCurrentIndex(1)

        mi = getattr(self._canvas, '_method_i', '?')
        mo = getattr(self._canvas, '_method_o', '?')
        self._lbl_mask_result.setText(
            f"✅ 추출 완료\n"
            f"   🔴 1차선: {ni}개  [{mi}]\n"
            f"   🔵 2차선: {no}개  [{mo}]\n"
            f"   ↓ 아래 '내보낼 차선' 선택 후 GT JSON 저장"
        )
        self._lbl_result.setText(
            f"✅ 수동 마스크 기반\n"
            f"   1차선(🔴): {ni}개\n"
            f"   2차선(🔵): {no}개"
        )
        # Step 3 그룹으로 사용자 주의 유도
        hint = ""
        if ni > 0 and no > 0:
            hint = "  → '내보낼 차선'에서 1차선/2차선 선택 후 저장!"
        elif ni > 0:
            hint = "  → '내보낼 차선: 1차선(inner)' 선택 후 저장!"
        elif no > 0:
            hint = "  → '내보낼 차선: 2차선(outer)' 선택 후 저장!"
        # 추출 완료 후 "둘 다 표시"로 자동 전환 (한쪽만 표시 설정이면 안보임)
        self._lane_combo.setCurrentIndex(0)   # 🔴🔵 둘 다 표시

        self._set_status(
            f"추출 완료!  1차선 {ni}개  2차선 {no}개{hint}  "
            f"⚠️ '수동 점만' 선택 시 0개 — 차선을 선택하세요.")

    def _copy_lane(self, lane):
        pts = self._canvas._pts_i if lane == "inner" else self._canvas._pts_o
        if not pts: self._set_status("먼저 Step 2를 실행하세요."); return
        self._canvas.set_manual(self._canvas.get_manual() + pts)
        self._update_count()
        self._set_status(f"{'1차선' if lane=='inner' else '2차선'} {len(pts)}개 → 수동 점에 복사 완료")

    # ── 검출 실행 ─────────────────────────────────────────────────────────────
    def _run_color(self):
        if self._canvas._src is None:
            self._set_status("이미지를 먼저 여세요."); return
        if self._seed_bgr is None:
            self._set_status("Step 1에서 도로 위를 클릭해 색상을 먼저 샘플하세요!"); return
        v = int(max(self._seed_bgr))
        if v < 90:
            if QMessageBox.question(
                self, "경고",
                f"샘플된 색상이 매우 어둡습니다 (V={v}).\n"
                "배경을 클릭했을 가능성이 높습니다.\n그래도 진행할까요?"
            ) != QMessageBox.Yes:
                return
        elif v > 200:
            if QMessageBox.question(
                self, "⚠️ 흰 줄 샘플됨",
                f"흰 차선 마킹이 샘플되었습니다 (V={v}).\n"
                "회색 도로 표면(V≈120)을 클릭해야 합니다.\n\n"
                "'V 범위로 검출+차선 분리' 버튼을 사용하면\n"
                "색상 샘플 없이도 정확하게 검출됩니다.\n\n"
                "그래도 현재 색으로 진행할까요?"
            ) != QMessageBox.Yes:
                return
        self._run("color")

    def _run_v(self):
        if self._canvas._src is None:
            self._set_status("이미지를 먼저 여세요."); return
        self._run("v_range")

    def _run(self, mode):
        self._set_status("검출 중...")
        self._lbl_result.setText("처리 중...")
        self._worker = Worker(
            self._canvas._src, mode=mode,
            seed_bgr=self._seed_bgr, tol=self._tol_sl.value(),
            v_min=self._sl_vmin.value(), v_max=self._sl_vmax.value(),
        )
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _on_done(self, result):
        if not result.get("ok"):
            msg = result.get("msg", "오류")
            self._set_status(f"❌ {msg}")
            self._lbl_result.setText(f"❌ {msg}")
            return

        self._canvas.set_result(
            result["road"], result["inner"], result["outer"],
            result["pts_i"], result["pts_o"], result["center"]
        )
        ni, no = len(result["pts_i"]), len(result["pts_o"])
        cx, cy = result["center"]
        br = result.get("boundary_r", 0.0)
        br_txt = (f"내부 타원 경계: {br:.0f}px ✅"
                  if br > 0 else "내부 타원 미검출 (전체 마스크 사용)")
        self._lbl_result.setText(
            f"✅ 도로: {result['n_road']:,}px\n"
            f"   {br_txt}\n"
            f"   1차선(🔴): {ni}개\n"
            f"   2차선(🔵): {no}개\n"
            f"   트랙 중심: ({cx},{cy})"
        )
        self._set_status(
            f"완료! 내부경계:{br:.0f}px  1차선:{ni}개  2차선:{no}개  "
            "— '내보낼 차선'을 선택하고 저장하세요.")

    # ── 편집 상태 저장/불러오기 ──────────────────────────────────────────────
    def _save_edit_state(self):
        state = self._canvas.get_edit_state()
        state["track_image"] = self._track_path
        try:
            with open(self._edit_state_path, "w") as f:
                json.dump(state, f, indent=2)
            ni_o = len(self._canvas._edit_poly_i_out)
            ni_i = len(self._canvas._edit_poly_i_inn)
            no_o = len(self._canvas._edit_poly_o_out)
            no_i = len(self._canvas._edit_poly_o_inn)
            nm   = len(self._canvas._manual)
            self._set_status(
                f"✅ 편집 상태 저장: 1차선({ni_o}/{ni_i}꼭짓점) "
                f"2차선({no_o}/{no_i}꼭짓점) 수동점({nm}개)  → {self._edit_state_path}")
        except Exception as e:
            self._set_status(f"❌ 저장 실패: {e}")

    def _load_edit_state(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "편집 상태 / GT JSON 불러오기",
            os.path.dirname(self._edit_state_path),
            "JSON (*.json)")
        if not p: return
        try:
            with open(p) as f:
                state = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "오류", f"파일 읽기 실패:\n{e}"); return

        if self._canvas._src is None:
            QMessageBox.warning(self, "경고",
                "이미지를 먼저 열어주세요.\n"
                f"저장된 이미지: {state.get('track_image', '알 수 없음')}")
            return

        # ── 포맷 자동 감지 ─────────────────────────────────────────────────
        # GT JSON 포맷: "centerline_pixels" 키가 있음
        if "centerline_pixels" in state:
            self._load_gt_json(state, p)
            return

        # 편집 상태 포맷: "edit_polygons" 키가 있음
        self._canvas.set_edit_state(state)
        self._update_count()

        ni_o = len(self._canvas._edit_poly_i_out)
        ni_i = len(self._canvas._edit_poly_i_inn)
        no_o = len(self._canvas._edit_poly_o_out)
        no_i = len(self._canvas._edit_poly_o_inn)
        nm   = len(self._canvas._manual)

        self._lbl_mask_result.setText(
            f"✅ 편집 상태 복원\n"
            f"   1차선 외곽:{ni_o} 내곽:{ni_i}꼭짓점\n"
            f"   2차선 외곽:{no_o} 내곽:{no_i}꼭짓점\n"
            f"   수동 점: {nm}개"
        )
        self._set_status(
            f"✅ 편집 상태 불러오기 완료! 마스크 복원됨.  "
            "'꼭짓점 편집 모드' 또는 '중앙선 추출' 바로 사용 가능.")

    def _load_gt_json(self, state: dict, path: str):
        """
        GT JSON 포맷 불러오기.
        centerline_pixels → 수동 점(_manual) 으로 로드하여 시각화.
        """
        pixels = state.get("centerline_pixels", [])
        if not pixels:
            QMessageBox.warning(self, "경고",
                f"centerline_pixels 가 비어 있습니다.\n{path}")
            return

        pts = [tuple(p) for p in pixels]
        lane = state.get("meta", {}).get("lane", "?")

        # 수동 점에 추가 (기존 점 대체 여부 질문)
        existing = self._canvas.get_manual()
        if existing:
            reply = QMessageBox.question(
                self, "기존 점 처리",
                f"기존 수동 점 {len(existing)}개가 있습니다.\n"
                "대체할까요? (아니오 = 추가)",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self._canvas.set_manual(pts)
            else:
                self._canvas.set_manual(existing + pts)
        else:
            self._canvas.set_manual(pts)

        self._update_count()
        nm = len(self._canvas.get_manual())

        # 차선 콤보 → 수동 점만 으로 전환 (gt json은 픽셀 점 그대로 표시)
        self._export_combo.setCurrentIndex(2)

        self._lbl_mask_result.setText(
            f"✅ GT JSON 불러오기 완료\n"
            f"   차선: {lane}\n"
            f"   포인트: {nm}개 → 수동 점으로 로드\n"
            f"   저장 시 '✏️ 수동 점만' 선택"
        )
        self._set_status(
            f"✅ GT JSON 불러오기: {os.path.basename(path)}  "
            f"차선={lane}  {nm}개 점 → 수동 점으로 로드됨")

    def _change_es_path(self):
        p, _ = QFileDialog.getSaveFileName(
            self, "편집 상태 저장 경로 변경",
            self._edit_state_path, "JSON (*.json)")
        if p:
            self._edit_state_path = p
            self._lbl_es_path.setText(f"경로:\n{p}")

    # ── 피처 주석 슬롯 ────────────────────────────────────────────────────────
    def _feat_summary(self):
        feats = self._canvas.get_features()
        if not feats:
            self._lbl_feat_result.setText("피처: 없음"); return
        cnt = {}
        for f in feats:
            cnt[f["kind"]] = cnt.get(f["kind"], 0) + 1
        order = ["vertical_parking", "parallel_parking", "in_line", "out_line",
                 "crosswalk", "start_point", "obstacle", "traffic_light"]
        lines = [f"  {k}: {cnt[k]}" for k in order if k in cnt]
        self._lbl_feat_result.setText("피처 합계:\n" + "\n".join(lines))

    def _load_auto_features(self):
        self._canvas.load_features(build_default_features())
        self._feat_summary()
        self._set_status("자동 검출 피처 로드 완료 — '피처 편집 모드'로 보정하세요.")

    def _toggle_feature_edit(self, checked):
        if checked:
            self._btn_feat_edit.setText("🖱️  피처 편집 모드 ON")
            if not self._canvas.get_features():
                self._load_auto_features()
            self._canvas.set_mode("feature")
            self._canvas.setFocus()
            # 다른 모드 버튼 해제
            self._btn_poly_add.setChecked(False)
            self._btn_poly_erase.setChecked(False)
            self._btn_poly_edit.setChecked(False)
            self._mode_bg.setExclusive(False)
            for rb in self._mode_bg.buttons():
                rb.setChecked(False)
            self._mode_bg.setExclusive(True)
            self._set_status("피처 편집: 코너/끝점 드래그=수정, 내부 드래그=이동, "
                             "출발점 우클릭=삭제")
        else:
            self._btn_feat_edit.setText("🖱️  피처 편집 모드 OFF")
            self._btn_feat_addstart.setChecked(False)
            self._canvas.set_mode("draw")
            self._mode_bg.buttons()[0].setChecked(True)

    def _toggle_add_start(self, checked):
        self._canvas.set_feat_add_start(checked)
        self._btn_feat_addstart.setText(
            "➕  출발점 추가 (클릭) ON" if checked else "➕  출발점 추가 (클릭) OFF")
        if checked and not self._btn_feat_edit.isChecked():
            self._btn_feat_edit.setChecked(True)   # 편집 모드 자동 진입

    def _save_features(self):
        feats = self._canvas.get_features()
        if not feats:
            QMessageBox.warning(self, "경고", "저장할 피처가 없습니다."); return
        groups = {}
        for f in feats:
            groups.setdefault(f["kind"], []).append(feature_to_world(f))
        out = {
            "meta": {
                "tool": "gt_annotator v4 · features",
                "source_png": self._track_path or TRACK_PNG_PATH,
                "formula": {"world_x": "-20.237+(py/884)*40.473",
                            "world_y": "-26.915+(px/1180)*53.83"},
                "wheel_fr_local": list(WHEEL_FR_LOCAL),
                "start_yaw": START_YAW,
                "spawn_note": "start_point.spawn_pose = 오른쪽 앞바퀴 중앙을 "
                              "point에 맞춘 base_link 스폰 pose (x,y,yaw).",
            },
            "features": groups,
        }
        with open(self._feat_path, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        self._feat_summary()
        QMessageBox.information(self, "저장 완료",
            f"피처 저장 완료!\n경로: {self._feat_path}\n총 {len(feats)}개")
        self._set_status(f"피처 저장: {len(feats)}개  ({self._feat_path})")

    def _load_features_json(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "피처 JSON 불러오기", self._feat_path, "JSON (*.json)")
        if not p:
            return
        try:
            with open(p) as f:
                data = json.load(f)
            feats = []
            for kind, items in data.get("features", {}).items():
                for d in items:
                    d.setdefault("kind", kind)
                    feats.append(feature_from_world(d))
            self._canvas.load_features(feats)
            self._feat_path = p
            self._lbl_feat_path.setText(f"경로:\n{p}")
            self._feat_summary()
            self._set_status(f"피처 불러오기: {len(feats)}개  ({os.path.basename(p)})")
        except Exception as ex:
            QMessageBox.warning(self, "오류", f"불러오기 실패:\n{ex}")

    def _change_feat_path(self):
        p, _ = QFileDialog.getSaveFileName(
            self, "피처 저장 경로 변경", self._feat_path, "JSON (*.json)")
        if p:
            self._feat_path = p
            self._lbl_feat_path.setText(f"경로:\n{p}")

    # ── 파일 I/O ──────────────────────────────────────────────────────────────
    def _open_dialog(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "트랙 이미지 열기", os.path.expanduser("~"),
            "Images (*.png *.jpg *.bmp)")
        if p: self._load(p)

    def _load(self, path):
        if self._canvas.load(path):
            self._track_path = path
            self._lbl_file.setText(os.path.basename(path))
            self._canvas.set_scale(self._zoom_sl.value() / 100.0)
            self._set_status(f"로드 완료: {path}")
        else:
            self._set_status(f"로드 실패: {path}")

    def _change_path(self):
        p, _ = QFileDialog.getSaveFileName(self, "저장 경로", self._gt_path, "JSON (*.json)")
        if p: self._gt_path = p; self._lbl_gt.setText(f"경로:\n{p}")

    def _get_pts(self):
        sel = self._export_combo.currentIndex()
        if sel == 2: return self._canvas.get_manual()
        return self._canvas.get_pts("inner" if sel == 0 else "outer")

    def _save_gt(self):
        pts = self._get_pts()
        if not pts: QMessageBox.warning(self, "경고", "포인트가 없습니다."); return
        world = [pixel_to_world(px, py) for px, py in pts]
        lane = ["inner", "outer", "manual"][self._export_combo.currentIndex()]
        out = {
            "meta": {
                "tool": "gt_annotator v4",
                "lane": lane,
                "formula": {"world_x": "-20.237+(py/884)*40.473",
                             "world_y": "-26.915+(px/1180)*53.83"},
                "landmarks": {k: list(v) for k, v in LANDMARKS.items()},
            },
            "centerline_pixels": [[int(x), int(y)] for x, y in pts],
            "centerline_world":  [[round(wx, 4), round(wy, 4)] for wx, wy in world],
        }
        with open(self._gt_path, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        QMessageBox.information(self, "저장 완료",
            f"GT 저장 완료!\n경로: {self._gt_path}\n포인트: {len(pts)}개  차선: {lane}")
        self._set_status(f"저장 완료: {len(pts)}개  ({lane})")

    def _export_py(self):
        pts = self._get_pts()
        if not pts: QMessageBox.warning(self, "경고", "포인트 없음"); return
        world = [pixel_to_world(px, py) for px, py in pts]
        td = "/home/autolab/VLA_simulation/src/mission_control_pkg/mission_control_pkg/track_data.py"
        if not os.path.exists(td):
            QMessageBox.warning(self, "오류", f"파일 없음:\n{td}"); return
        with open(td) as f: content = f.read()
        lines = "\n".join(f"    ({wx:.4f}, {wy:.4f})," for wx, wy in world)
        new_block = (
            "# 🤖 자동 생성 GT 중앙선 (gt_annotator)\n"
            f"TRACK_CENTERLINE: list[tuple[float, float]] = [\n{lines}\n]\n"
        )
        import re
        # 줄 시작에 TRACK_CENTERLINE이 있는 변수 정의만 교체
        # (for/if 문 내부의 TRACK_CENTERLINE 참조는 제외)
        patt = r'^TRACK_CENTERLINE\s*(?::\s*[^\n=]*)?\s*=\s*\[[^\]]*?\]\s*\n'
        content = (re.sub(patt, new_block, content, flags=re.MULTILINE | re.DOTALL)
                   if re.search(patt, content, flags=re.MULTILINE | re.DOTALL)
                   else content + "\n" + new_block)
        with open(td, "w") as f: f.write(content)
        QMessageBox.information(self, "완료",
            f"TRACK_CENTERLINE 업데이트!\n{len(world)}개 포인트\n{td}")
        self._set_status(f"track_data.py 업데이트 완료: {len(world)}개")


def main(args=None):
    app = QApplication.instance() or QApplication(sys.argv)
    win = GTAnnotator(); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

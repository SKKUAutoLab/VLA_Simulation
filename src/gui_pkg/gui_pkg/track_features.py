#!/usr/bin/env python3
"""
track_features.py
=================
track.png 에서 차선(중앙선) 이외의 정적 트랙 피처를 수치화한다.

추출 대상:
  - vertical_parking   : 수직 주차칸 4칸 (남향 개방)
  - parallel_parking   : 평행 주차칸 4칸
  - in_line / out_line : 인필드 진입(IN) / 진출(OUT) 정지선
  - crosswalk          : 횡단보도(우측 zebra)

좌표 변환 (gt_annotator.py 와 동일):
  world_x = -20.237 + (py / 884) * 40.473
  world_y = -26.915 + (px / 1180) * 53.83
"""
import json, os, math
import numpy as np
import cv2

IMG_W, IMG_H = 1180.0, 884.0
TRACK_PNG = ("/home/autolab/ros2_autonomous_vehicle_simulation/src/simulation_pkg/"
             "models/race_track/materials/textures/track.png")
OUT_JSON = os.path.expanduser("~/track_features.json")

# 차량(prius_hybrid) 앞우측 휠 중심 — 모델 로컬좌표 (전방 = 모델 -y, 좌측 = +x)
WHEEL_FR_LOCAL = (-0.760002, -1.41)
# IN 출발 heading: '← IN' 방향(서쪽). yaw=0 → 모델 -y가 월드 -y(서쪽)를 향함
START_YAW = 0.0


def p2w(px, py):
    """픽셀 (px=열, py=행) → 월드 (x, y)."""
    return (-20.237 + (py / IMG_H) * 40.473,
            -26.915 + (px / IMG_W) * 53.83)


def spawn_pose_from_dot(px, py, yaw=START_YAW):
    """
    오른쪽 앞바퀴 중앙이 픽셀 점 (px,py)에 오도록 하는 base_link 스폰 pose.

      world_dot = spawn_origin + Rz(yaw) · wheel_local
      → spawn_origin = world_dot − Rz(yaw) · wheel_local

    반환: (x, y, yaw)  — gazebo spawn_entity 의 -x -y -Y 에 그대로 사용.
    """
    wx, wy = p2w(px, py)
    lx, ly = WHEEL_FR_LOCAL
    c, s = math.cos(yaw), math.sin(yaw)
    ox = wx - (c * lx - s * ly)
    oy = wy - (s * lx + c * ly)
    return ox, oy, yaw


def w2p(wx, wy):
    """월드 (x,y) → 픽셀 (px, py)."""
    return (int((wy + 26.915) / 53.83 * IMG_W),
            int((wx + 20.237) / 40.473 * IMG_H))


def dot_from_spawn(sx, sy, yaw):
    """base_link 스폰 pose → 오른쪽 앞바퀴 중앙 픽셀점."""
    lx, ly = WHEEL_FR_LOCAL
    c, s = math.cos(yaw), math.sin(yaw)
    return w2p(sx + (c * lx - s * ly), sy + (s * lx + c * ly))


def parking_target_poses(kind, cx, cy):
    """주차칸 중심(world) → 전면/후면 주차 목표 base_link pose."""
    if kind == "vertical_parking":
        fy, ry = -math.pi / 2, math.pi / 2
    else:
        fy, ry = 0.0, math.pi
    return {
        "front": {"desc": "전면주차(nose-in)",
                  "base_pose": {"x": round(cx, 4), "y": round(cy, 4), "yaw": round(fy, 6)}},
        "rear":  {"desc": "후면주차(back-in)",
                  "base_pose": {"x": round(cx, 4), "y": round(cy, 4), "yaw": round(ry, 6)}},
    }


# ── 정적 객체 (012_deploy_lib.py, Gazebo 월드좌표) ────────────────────────────
TRAFFIC_LIGHTS = [
    {"id": "TL1", "world": (-5.6255, 17.9036), "yaw": 1.568773, "src": "traffic_light_stand"},
]
OBSTACLES = [
    {"id": "OB_fix1", "world": (-3.659642, 8.710748),   "yaw": -0.013934, "src": "obstacle_coordinates_1"},
    {"id": "OB_fix2", "world": (-3.659642, 2.037476),   "yaw": -0.013934, "src": "obstacle_coordinates_2"},
    {"id": "OB_av1",  "world": (12.251981, -15.909271), "yaw": 2.484252,  "src": "obstacle_coordinates1"},
    {"id": "OB_p2a",  "world": (11.884767, 11.605120),  "yaw": 3.25, "src": "obstacle_coordinates2"},
    {"id": "OB_p2b",  "world": (12.040719, 10.060495),  "yaw": 3.25, "src": "obstacle_coordinates2"},
    {"id": "OB_p2c",  "world": (12.230394, 8.181866),   "yaw": 3.25, "src": "obstacle_coordinates2"},
    {"id": "OB_p3a",  "world": (16.106836, -0.111269),  "yaw": 3.25, "src": "obstacle_coordinates3"},
    {"id": "OB_p3b",  "world": (16.281788, -1.844067),  "yaw": 3.25, "src": "obstacle_coordinates3"},
    {"id": "OB_p3c",  "world": (16.366463, -3.446680),  "yaw": 3.25, "src": "obstacle_coordinates3"},
]
OUT_START_SPAWNS = [
    (-1.672862, -16.311572, -3.133789),
    (-1.681217, -15.244810, -3.133795),
    (-0.772971, -15.237810, -3.133797),
    (-0.764668, -16.302988, -3.133797),
]


def rect_world(x0, y0, x1, y1, kind=None):
    """픽셀 사각형 → 중심 월드좌표 + 월드 크기 + 4코너 (+주차목표pose)."""
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    wx, wy = p2w(cx, cy)
    # 월드 치수: x(전진)는 py(세로)에, y(횡)는 px(가로)에 대응
    size_x = abs(y1 - y0) / IMG_H * 40.473   # 세로 길이 → world_x 방향
    size_y = abs(x1 - x0) / IMG_W * 53.83    # 가로 길이 → world_y 방향
    corners = [list(p2w(x, y)) for x, y in
               [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]
    d = {
        "center_pixel":  [round(cx, 1), round(cy, 1)],
        "center_world":  [round(wx, 4), round(wy, 4)],
        "size_world":    {"x": round(size_x, 4), "y": round(size_y, 4)},
        "bbox_pixel":    [int(x0), int(y0), int(x1), int(y1)],
        "corners_world": [[round(a, 4), round(b, 4)] for a, b in corners],
    }
    if kind in ("vertical_parking", "parallel_parking"):
        d["target_poses"] = parking_target_poses(kind, wx, wy)
    return d


def build():
    # ── 측정된 픽셀 기하 (track.png 투영 분석 결과) ──────────────────────────
    # 수직 주차칸: 분리선 x, 상단 y=311, 하단 y=425 (남향 개방)
    V_DIV = [521, 596, 670, 743, 817]
    V_TOP, V_BOT = 311, 425
    # 평행 주차칸: 분리선 x, 상단 y=565, 하단 y=632
    P_DIV = [364, 478, 592, 707, 822]
    P_TOP, P_BOT = 565, 632

    features = {}

    features["vertical_parking"] = [
        {"id": f"V{i+1}", **rect_world(V_DIV[i], V_TOP, V_DIV[i+1], V_BOT,
                                       "vertical_parking")}
        for i in range(len(V_DIV) - 1)
    ]
    features["parallel_parking"] = [
        {"id": f"P{i+1}", **rect_world(P_DIV[i], P_TOP, P_DIV[i+1], P_BOT,
                                       "parallel_parking")}
        for i in range(len(P_DIV) - 1)
    ]

    # IN/OUT 정지선: 짧은 세로 흰 바
    features["out_line"] = {
        "desc": "인필드 진출(OUT) 정지선, 'OUT ←'", **rect_world(284, 440, 288, 548)}
    features["in_line"] = {
        "desc": "인필드 진입(IN) 정지선, '← IN'", **rect_world(893, 440, 897, 549)}

    # 횡단보도: 우측 zebra (세로 줄무늬 6개), x 1010..1120, y 340..409
    features["crosswalk"] = {
        "desc": "우측 횡단보도(zebra)", "n_stripes": 6,
        **rect_world(1010, 340, 1120, 409)}

    # 출발점: IN측(텍스처 점, yaw=0=서쪽) + OUT측(parking_start 역산, yaw≈-π=동쪽)
    features["start_point"] = []

    def _add_start(sid, side, px, py, yaw):
        wx, wy = p2w(px, py)
        ox, oy, oyaw = spawn_pose_from_dot(px, py, yaw)
        features["start_point"].append({
            "id": sid, "side": side,
            "point_pixel": [int(px), int(py)],
            "point_world": [round(wx, 4), round(wy, 4)],
            "yaw": round(yaw, 6),
            "spawn_pose": {"x": round(ox, 4), "y": round(oy, 4), "yaw": round(oyaw, 6)},
            "desc": "오른쪽 앞바퀴 중앙=point. spawn_pose=base_link 스폰값.",
        })

    for i, (px, py) in enumerate([(914, 483), (938, 483), (914, 508), (938, 508)]):
        _add_start(f"IN{i+1}", "IN", px, py, START_YAW)
    for i, (sx, sy, syaw) in enumerate(OUT_START_SPAWNS):
        dpx, dpy = dot_from_spawn(sx, sy, syaw)
        _add_start(f"OUT{i+1}", "OUT", dpx, dpy, syaw)

    # 장애물 / 신호등 (월드좌표 → 픽셀)
    def _obj(o):
        px, py = w2p(*o["world"])
        return {"id": o["id"], "point_pixel": [px, py],
                "point_world": [round(o["world"][0], 4), round(o["world"][1], 4)],
                "yaw": round(o["yaw"], 6), "src": o["src"],
                "object_pose": {"x": round(o["world"][0], 4),
                                "y": round(o["world"][1], 4),
                                "yaw": round(o["yaw"], 6)}}
    features["obstacle"] = [_obj(o) for o in OBSTACLES]
    features["traffic_light"] = [_obj(o) for o in TRAFFIC_LIGHTS]

    return features


def main():
    features = build()
    out = {
        "meta": {
            "tool": "track_features.py",
            "source_png": TRACK_PNG,
            "formula": {"world_x": "-20.237+(py/884)*40.473",
                        "world_y": "-26.915+(px/1180)*53.83"},
            "wheel_fr_local": list(WHEEL_FR_LOCAL),
            "start_yaw": START_YAW,
            "spawn_note": "start_point.spawn_pose = 오른쪽 앞바퀴 중앙을 point에 "
                          "맞춘 base_link 스폰 pose. parking target_poses = "
                          "전면/후면 주차 목표 base_link pose.",
        },
        "features": features,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # 콘솔 요약
    print(f"저장: {OUT_JSON}\n")
    for kind in ("vertical_parking", "parallel_parking"):
        print(f"[{kind}]")
        for s in features[kind]:
            w = s["center_world"]; sz = s["size_world"]
            print(f"  {s['id']}: center=({w[0]:+.3f}, {w[1]:+.3f})  "
                  f"size={sz['x']:.2f}x{sz['y']:.2f} m  px{s['bbox_pixel']}")
        print()
    for kind in ("in_line", "out_line", "crosswalk"):
        f_ = features[kind]; w = f_["center_world"]
        print(f"[{kind}] center=({w[0]:+.3f}, {w[1]:+.3f})  {f_['desc']}")
    print("\n[start_point]  (오른쪽 앞바퀴 중앙 정렬)")
    for s in features["start_point"]:
        w = s["point_world"]; sp = s["spawn_pose"]
        print(f"  {s['id']} [{s['side']}]: point=({w[0]:+.3f}, {w[1]:+.3f})  "
              f"spawn=({sp['x']:+.3f}, {sp['y']:+.3f}, yaw={sp['yaw']:+.3f})")
    print("\n[obstacle / traffic_light]")
    for s in features["obstacle"] + features["traffic_light"]:
        w = s["point_world"]
        print(f"  {s['id']}: ({w[0]:+.3f}, {w[1]:+.3f}, yaw={s['yaw']:+.3f})  [{s['src']}]")
    print("\n[parking target_poses] 예시 V1:",
          features["vertical_parking"][0]["target_poses"])


if __name__ == "__main__":
    main()

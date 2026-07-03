#!/usr/bin/env python3
"""
gt_extractor.py
===============
탑 카메라 이미지에서 트랙 도로 중앙선 GT를 자동 추출하는 도구

동작 방식:
  1. /top_camera/image_raw 구독 → 이미지 캡처
  2. 도로 색상 임계값으로 마스크 생성 (도로는 밝은 회색, 배경은 어두운 색)
  3. 마스크에서 중앙선 스켈레톤 추출
  4. 픽셀 좌표 → Gazebo 세계 좌표 변환
  5. 결과를 JSON 파일로 저장 + 시각화

카메라 파라미터 (track.world):
  - 위치: (0, 0, 41.619)  pitch=1.57 (아래 방향)
  - FOV: horizontal = 1.0856 rad
  - 이미지 크기: 640x480

실행:
  ros2 run gui_pkg gt_extractor
"""

import sys
import json
import math
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSHistoryPolicy,
                        QoSDurabilityPolicy, QoSReliabilityPolicy)
from sensor_msgs.msg import Image

# ─── 카메라 파라미터 ─────────────────────────────────────────────────────
# track.world 기준
CAM_HEIGHT   = 41.619      # 카메라 높이 [m]
CAM_X        = 0.0         # 카메라 세계 X
CAM_Y        = 0.0         # 카메라 세계 Y
H_FOV        = 1.0856      # 수평 FOV [rad]
IMG_W        = 640
IMG_H        = 480

# 픽셀당 미터 계산
half_fov     = H_FOV / 2.0
world_half_w = CAM_HEIGHT * math.tan(half_fov)   # 약 25.14m
world_half_h = world_half_w * (IMG_H / IMG_W)    # 약 18.86m

PX_PER_M     = IMG_W / (2 * world_half_w)         # 약 12.73 px/m

# 픽셀 → 세계 좌표 변환 함수
# 주의: 카메라 방향에 따라 축 매핑이 결정됨
# 탑 카메라 pitch=1.57, yaw=0 → image right = world +X, image down = world -Y
# (Gazebo 시뮬레이터 실행 후 실제 확인 필요)
def pixel_to_world(px, py):
    """
    이미지 픽셀 (px, py) → Gazebo 세계 좌표 (wx, wy)

    중심 픽셀 (320, 240) → 세계 (0, 0)
    """
    wx = CAM_X + (px - IMG_W / 2) / PX_PER_M
    wy = CAM_Y - (py - IMG_H / 2) / PX_PER_M   # Y축 반전 (이미지 down = 세계 -Y)
    return wx, wy


def world_to_pixel(wx, wy):
    """Gazebo 세계 좌표 → 이미지 픽셀."""
    px = int((wx - CAM_X) * PX_PER_M + IMG_W / 2)
    py = int(-(wy - CAM_Y) * PX_PER_M + IMG_H / 2)
    return px, py


# ─── 알려진 GT 좌표로 축 방향 보정 ────────────────────────────────────────
def calibrate_axis_mapping(image: np.ndarray) -> str:
    """
    알려진 랜드마크 위치로 카메라 축 방향을 확인합니다.
    Returns: "standard" (image right=+X, down=-Y) 또는 "rotated"
    """
    # 알려진 장애물 좌표들을 픽셀로 변환하고 해당 픽셀 색상 확인
    known_points = {
        "obstacle_1": (-3.66, 8.71),
        "obstacle_2": (-3.66, 2.04),
        "start":      (-2.55, -22.71),
    }

    info = {}
    for name, (wx, wy) in known_points.items():
        px, py = world_to_pixel(wx, wy)
        if 0 <= px < IMG_W and 0 <= py < IMG_H:
            info[name] = (px, py, image[py, px].tolist())

    return info


# ─── 도로 세그멘테이션 ────────────────────────────────────────────────────
def segment_road(image: np.ndarray) -> np.ndarray:
    """
    탑 카메라 이미지에서 도로 영역을 마스크로 추출.

    트랙 도로 특징:
    - 회색 아스팔트 색상 (race_track 텍스처)
    - 배경보다 밝음
    """
    # HSV 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 방법 1: 밝기(Saturation 낮음 + 적당한 밝기)로 도로 검출
    # 도로: saturation 낮고, value 중간~높음
    lower_road = np.array([0,   0,  80])   # HSV 하한
    upper_road = np.array([180, 60, 220])  # HSV 상한
    mask_hsv = cv2.inRange(hsv, lower_road, upper_road)

    # 방법 2: Canny edge 후 도로 내부 채우기
    edges = cv2.Canny(gray, 30, 80)

    # 노이즈 제거
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask_clean, edges


def extract_centerline_from_mask(mask: np.ndarray, sample_step: int = 20) -> list:
    """
    도로 마스크에서 중앙선 포인트를 추출합니다.

    방법:
    1. 각 행/열에서 도로 픽셀의 중앙점 계산
    2. 세계 좌표로 변환
    """
    centerline_px = []

    # Y방향 슬라이딩 (행 단위)
    for py in range(0, IMG_H, sample_step):
        row = mask[py, :]
        road_pixels = np.where(row > 128)[0]

        if len(road_pixels) == 0:
            continue

        # 연속된 그룹으로 분리
        groups = []
        start = road_pixels[0]
        prev  = road_pixels[0]
        for px in road_pixels[1:]:
            if px - prev > 10:  # 10픽셀 이상 갭이면 새 그룹
                groups.append((start, prev))
                start = px
            prev = px
        groups.append((start, prev))

        # 각 그룹의 중앙점
        for gs, ge in groups:
            if ge - gs < 5:  # 너무 작은 그룹 제외
                continue
            cx = (gs + ge) // 2
            centerline_px.append((cx, py))

    return centerline_px


def pixels_to_world_coords(px_pts: list) -> list:
    """픽셀 좌표 리스트 → 세계 좌표 리스트."""
    return [pixel_to_world(px, py) for px, py in px_pts]


def filter_and_sort_centerline(world_pts: list) -> list:
    """중앙선 포인트를 필터링하고 순서 정렬."""
    if not world_pts:
        return []

    pts = np.array(world_pts)

    # Y 기준 정렬 (스타트 포인트부터)
    # 우선 가장 가까운 점부터 연결
    sorted_pts = []
    remaining = list(range(len(pts)))

    # 시작점: y값이 가장 작은 점 (스폰 근처)
    start_idx = np.argmin(pts[:, 1])
    current_idx = start_idx

    while remaining:
        sorted_pts.append(pts[current_idx].tolist())
        remaining.remove(current_idx)

        if not remaining:
            break

        # 가장 가까운 미방문 포인트 찾기
        dists = [
            math.hypot(pts[i][0] - pts[current_idx][0],
                       pts[i][1] - pts[current_idx][1])
            for i in remaining
        ]
        nearest = remaining[np.argmin(dists)]

        if min(dists) > 10.0:  # 10m 이상 점프하면 끊어진 것
            break

        current_idx = nearest

    return sorted_pts


def visualize_extraction(image, mask, centerline_px, centerline_world):
    """추출 결과 시각화."""
    vis = image.copy()

    # 마스크 오버레이 (반투명 초록)
    overlay = vis.copy()
    overlay[mask > 128] = [0, 200, 0]
    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

    # 중앙선 포인트 (빨간 점)
    for px, py in centerline_px:
        cv2.circle(vis, (px, py), 3, (0, 0, 255), -1)

    # 알려진 랜드마크 표시 (파란 원)
    landmarks = {
        "Start":    (-2.55, -22.71),
        "TL":       (-5.63,  17.90),
        "Obs1":     (-3.66,   8.71),
        "Obs2":     (-3.66,   2.04),
    }
    for name, (wx, wy) in landmarks.items():
        px, py = world_to_pixel(wx, wy)
        if 0 <= px < IMG_W and 0 <= py < IMG_H:
            cv2.circle(vis, (px, py), 8, (255, 0, 0), 2)
            cv2.putText(vis, name, (px+5, py-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

    # 카메라 파라미터 표시
    info_text = [
        f"CAM: (0, 0, {CAM_HEIGHT}m)  FOV:{math.degrees(H_FOV):.1f}deg",
        f"Scale: {PX_PER_M:.2f} px/m",
        f"World range X: {-world_half_w:.1f}~{world_half_w:.1f}m",
        f"World range Y: {-world_half_h:.1f}~{world_half_h:.1f}m",
        f"Extracted points: {len(centerline_world)}",
    ]
    for i, t in enumerate(info_text):
        cv2.putText(vis, t, (5, 15 + i*16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)

    return vis


# ─── ROS 노드 ─────────────────────────────────────────────────────────────
class GTExtractorNode(Node):
    def __init__(self):
        super().__init__("gt_extractor_node")

        self._image_received = False
        self._image          = None

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        # 탑 카메라 구독 시도
        self.create_subscription(Image, "/top_camera/image_raw",
                                 self._img_cb, qos)
        self.get_logger().info(
            "GT 추출기 시작\n"
            "  /top_camera/image_raw 구독 중...\n"
            "  이미지 수신 후 자동으로 GT 추출 시작"
        )

    def _img_cb(self, msg: Image):
        if self._image_received:
            return
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        self._image = arr.reshape((msg.height, msg.width, 3))
        if msg.encoding == 'bgr8':
            pass  # already BGR for OpenCV
        elif msg.encoding == 'rgb8':
            self._image = self._image[:, :, ::-1].copy()
        self._image_received = True
        self.get_logger().info("이미지 수신! 추출 시작...")
        self._extract()

    def _extract(self):
        img = self._image

        # 1. 교정 정보 출력
        calib = calibrate_axis_mapping(img)
        self.get_logger().info(f"알려진 좌표 확인:\n{json.dumps(calib, indent=2)}")

        # 2. 도로 세그멘테이션
        mask, edges = segment_road(img)

        # 3. 중앙선 추출
        centerline_px = extract_centerline_from_mask(mask, sample_step=15)
        centerline_world = pixels_to_world_coords(centerline_px)
        centerline_sorted = filter_and_sort_centerline(centerline_world)

        self.get_logger().info(
            f"추출된 중앙선 포인트: {len(centerline_sorted)}개"
        )

        # 4. 결과 저장
        output = {
            "camera": {
                "height": CAM_HEIGHT,
                "px_per_m": PX_PER_M,
                "world_range": {
                    "x": [-world_half_w, world_half_w],
                    "y": [-world_half_h, world_half_h],
                }
            },
            "centerline": centerline_sorted,
            "landmarks": {
                "start":       [-2.55, -22.71],
                "traffic_light": [-5.63, 17.90],
                "obstacle_1":  [-3.66,  8.71],
                "obstacle_2":  [-3.66,  2.04],
            },
            "note": (
                "world_X: image right = +X, "
                "world_Y: image down = -Y (확인 필요)"
            )
        }

        import os
        out_path = os.path.expanduser("~/track_gt.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        self.get_logger().info(f"GT 저장 완료: {out_path}")

        # 5. 시각화 저장
        vis = visualize_extraction(img, mask, centerline_px, centerline_sorted)
        vis_path = os.path.expanduser("~/track_gt_vis.png")
        cv2.imwrite(vis_path, vis)

        # 원본, 마스크, 엣지도 저장
        cv2.imwrite(os.path.expanduser("~/track_raw.png"), img)
        cv2.imwrite(os.path.expanduser("~/track_mask.png"), mask)

        self.get_logger().info(
            f"시각화 저장:\n"
            f"  원본:  ~/track_raw.png\n"
            f"  마스크: ~/track_mask.png\n"
            f"  결과:  ~/track_gt_vis.png\n\n"
            f"⚠️  축 방향 확인 필수:\n"
            f"  ~/track_gt_vis.png 에서 파란 원(알려진 랜드마크)이\n"
            f"  실제 도로 위에 표시되는지 확인하세요."
        )

        # 샘플 출력
        if centerline_sorted:
            sample = centerline_sorted[::max(1, len(centerline_sorted)//10)]
            self.get_logger().info(
                f"샘플 좌표:\n" +
                "\n".join(f"  ({x:.2f}, {y:.2f})" for x, y in sample)
            )

        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = GTExtractorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
parking_gt_extractor.py
=======================
parking.world TOP 카메라 이미지에서 OUT 선, IN 마크 위치를 자동 추출

실행:
  ros2 run gui_pkg parking_gt_extractor
"""

import os
import json
import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import Image

# ─── 카메라 파라미터 (parking.world) ─────────────────────────────────────────
CAM_HEIGHT = 40.619      # parking.world 카메라 높이 [m]
CAM_X      = 0.0
CAM_Y      = 0.0
H_FOV      = 1.0856      # 수평 FOV [rad]
IMG_W      = 640
IMG_H      = 480

half_fov     = H_FOV / 2.0
world_half_w = CAM_HEIGHT * math.tan(half_fov)
world_half_h = world_half_w * (IMG_H / IMG_W)
PX_PER_M     = IMG_W / (2 * world_half_w)


def pixel_to_world(px, py):
    wx = CAM_X + (px - IMG_W / 2) / PX_PER_M
    wy = CAM_Y - (py - IMG_H / 2) / PX_PER_M
    return wx, wy


def world_to_pixel(wx, wy):
    px = int((wx - CAM_X) * PX_PER_M + IMG_W / 2)
    py = int(-(wy - CAM_Y) * PX_PER_M + IMG_H / 2)
    return px, py


def detect_white_lines(image: np.ndarray):
    """
    흰색 OUT 선 검출.
    반환: 검출된 선 세그먼트 리스트 [(x1,y1,x2,y2), ...]
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 흰색 마스크: 밝기 200 이상, 채도 낮은 픽셀
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv,
                              np.array([0,   0, 180]),
                              np.array([180, 50, 255]))

    # 노이즈 제거
    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Hough 선 검출 (긴 수직/수평 선만)
    lines = cv2.HoughLinesP(white_mask, 1, np.pi / 180,
                             threshold=50,
                             minLineLength=30,
                             maxLineGap=10)

    result = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.hypot(x2 - x1, y2 - y1)
            if length > 30:  # 30px 이상만
                result.append((x1, y1, x2, y2, length))

    # 길이 내림차순 정렬
    result.sort(key=lambda x: x[4], reverse=True)
    return result


def extract_out_line_world(lines):
    """
    검출된 선 중 OUT 선 후보를 세계 좌표로 변환.
    가장 긴 선을 OUT 선으로 간주.
    """
    if not lines:
        return None

    out_lines = []
    for x1, y1, x2, y2, length in lines[:5]:  # 상위 5개
        wx1, wy1 = pixel_to_world(x1, y1)
        wx2, wy2 = pixel_to_world(x2, y2)
        cx = (wx1 + wx2) / 2
        cy = (wy1 + wy2) / 2
        angle = math.degrees(math.atan2(wy2 - wy1, wx2 - wx1))
        world_length = math.hypot(wx2 - wx1, wy2 - wy1)
        out_lines.append({
            "p1": [round(wx1, 4), round(wy1, 4)],
            "p2": [round(wx2, 4), round(wy2, 4)],
            "center": [round(cx, 4), round(cy, 4)],
            "angle_deg": round(angle, 2),
            "length_m": round(world_length, 3),
            "pixel": [x1, y1, x2, y2],
        })

    return out_lines


def visualize(image, lines, out_lines):
    vis = image.copy()

    # 흰색 마스크 오버레이
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv,
                              np.array([0,   0, 180]),
                              np.array([180, 50, 255]))
    overlay = vis.copy()
    overlay[white_mask > 0] = [0, 255, 255]  # 노란색으로 강조
    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

    # 검출된 선 표시
    colors = [(0, 0, 255), (0, 165, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]
    if out_lines:
        for i, ol in enumerate(out_lines[:5]):
            x1, y1, x2, y2 = ol["pixel"]
            color = colors[i % len(colors)]
            cv2.line(vis, (x1, y1), (x2, y2), color, 2)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            label = f"#{i+1} {ol['length_m']:.1f}m"
            cv2.putText(vis, label, (cx + 5, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # 기존 GT 주차 좌표 표시 (참고용)
    known = {
        "수직1": (-3.287, 8.656),
        "수직2": (-3.198, 5.280),
        "수직3": (-3.118, 1.906),
        "수직4": (-3.173, -1.437),
        "평행1": (7.121, 7.940),
        "평행2": (7.108, 2.732),
        "평행3": (7.103, -2.459),
        "평행4": (7.123, -7.549),
    }
    for name, (wx, wy) in known.items():
        px, py = world_to_pixel(wx, wy)
        if 0 <= px < IMG_W and 0 <= py < IMG_H:
            cv2.circle(vis, (px, py), 5, (255, 200, 0), -1)
            cv2.putText(vis, name, (px + 4, py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1)

    return vis


class ParkingGTExtractorNode(Node):
    def __init__(self):
        super().__init__("parking_gt_extractor_node")
        self._done = False

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.create_subscription(Image, "/top_camera/image_raw", self._cb, qos)
        self.get_logger().info("parking GT 추출기 시작 — /top_camera/image_raw 대기 중...")

    def _cb(self, msg: Image):
        if self._done:
            return
        self._done = True

        arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        image = arr.reshape((msg.height, msg.width, 3))
        if msg.encoding == 'rgb8':
            image = image[:, :, ::-1].copy()

        self.get_logger().info("이미지 수신! OUT 선 검출 시작...")

        # 흰색 선 검출
        lines = detect_white_lines(image)
        out_lines = extract_out_line_world(lines)

        # 결과 출력
        if out_lines:
            self.get_logger().info(f"검출된 선 {len(out_lines)}개:")
            for i, ol in enumerate(out_lines):
                self.get_logger().info(
                    f"  #{i+1}: 길이={ol['length_m']:.2f}m, "
                    f"center={ol['center']}, "
                    f"p1={ol['p1']}, p2={ol['p2']}, "
                    f"각도={ol['angle_deg']:.1f}°"
                )
        else:
            self.get_logger().warn("흰색 선을 찾지 못했습니다!")

        # JSON 저장
        out_path = os.path.expanduser("~/parking_gt_lines.json")
        with open(out_path, "w") as f:
            json.dump({
                "camera": {
                    "height": CAM_HEIGHT,
                    "px_per_m": round(PX_PER_M, 4),
                    "world_range": {
                        "x": [-round(world_half_w, 3), round(world_half_w, 3)],
                        "y": [-round(world_half_h, 3), round(world_half_h, 3)],
                    }
                },
                "detected_lines": out_lines,
            }, f, indent=2)
        self.get_logger().info(f"결과 저장: {out_path}")

        # 시각화 저장
        vis = visualize(image, lines, out_lines)
        vis_path = os.path.expanduser("~/parking_gt_vis.png")
        raw_path = os.path.expanduser("~/parking_raw.png")
        cv2.imwrite(vis_path, vis)
        cv2.imwrite(raw_path, image)
        self.get_logger().info(
            f"시각화: ~/parking_gt_vis.png\n"
            f"원본:   ~/parking_raw.png\n\n"
            f"⚠️  ~/parking_gt_vis.png 에서 OUT 선(빨간색 #1)이\n"
            f"   실제 OUT 선 위에 표시되는지 확인하세요!"
        )

        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = ParkingGTExtractorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()

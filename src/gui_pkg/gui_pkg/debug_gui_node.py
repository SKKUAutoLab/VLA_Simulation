#!/usr/bin/env python3
"""
debug_gui_node.py
=================
트랙 디버깅 GUI

트랙 위의 모든 정보를 실시간으로 시각화합니다:
  - 🗺 트랙 맵: 자차 위치, LiDAR 포인트, YOLOv8 탐지 객체, 경로, 웨이포인트
  - 📷 탑 카메라: /top_camera/image_raw 실제 영상
  - 📡 LiDAR 스캔: 극좌표(polar) 시각화
  - 실시간 사이드 패널: 차량/센서/신호등/차선/제어 상태

실행:
  ros2 run gui_pkg debug_gui_node
"""

import sys
import json
import math
import threading
import time
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSHistoryPolicy,
                        QoSDurabilityPolicy, QoSReliabilityPolicy)

from std_msgs.msg import String, Bool
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan
from interfaces_pkg.msg import (DetectionArray, LaneInfo,
                                  MotionCommand, PathPlanningResult)

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFrame, QTabWidget,
    QScrollArea, QSizePolicy, QGroupBox, QGridLayout,
    QTableWidget, QTableWidgetItem, QHeaderView,
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QPointF, QRectF, Slot
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPainterPath,
    QImage, QPixmap, QPainterPath, QPolygonF,
)

# ─── 트랙 GT 데이터 ───────────────────────────────────────────────────────
TRACK_CENTERLINE = [
    (-2.55, -22.71), (-2.55, -20.00), (-2.80, -17.00), (-3.00, -14.00),
    (-3.10, -10.00), (-3.20,  -6.00), (-3.40,  -2.00), (-3.66,   2.04),
    (-3.66,   5.00), (-3.66,   8.71), (-3.80,  11.00), (-4.20,  13.50),
    (-4.80,  15.50), (-5.63,  17.90), (-5.50,  19.50), (-3.00,  21.50),
    ( 0.00,  22.80), ( 3.50,  23.00), ( 7.00,  22.50), (10.50,  21.50),
    (12.50,  19.50), (13.50,  16.00), (14.00,  12.00), (13.50,   8.50),
    (12.50,   5.00), (16.00,   1.00), (16.37,  -2.00), (16.00,  -5.00),
    (14.50,  -9.00), (12.25, -13.00), (12.25, -15.91), (10.50, -19.50),
    ( 7.50, -22.50), ( 4.00, -23.50), ( 0.00, -23.50), (-2.55, -22.71),
]

LANDMARKS = {
    "출발점":        (-2.55, -22.71),
    "신호등/횡단보도": (-5.63,  17.90),
    "장애물구역-1":   (-3.66,   8.71),
    "장애물구역-2":   (-3.66,   2.04),
}

TRACK_BOUNDS = {"x_min": -8.0, "x_max": 20.0, "y_min": -26.0, "y_max": 25.0}
LANE_WIDTH_PX_SCALE = 2.8   # m


# ─── ROS 브리지 ───────────────────────────────────────────────────────────
class DebugBridge(QObject):
    odom_recv      = Signal(float, float, float)          # x, y, yaw_deg
    lidar_recv     = Signal(object, float, float)          # ranges_arr, angle_min, angle_inc
    obstacle_recv  = Signal(bool, float)                   # detected, min_dist
    detection_recv = Signal(object)                        # DetectionArray
    lane_recv      = Signal(float, object)                 # slope, target_points[]
    traffic_recv   = Signal(str)                           # color string
    path_recv      = Signal(object, object)                # x_arr, y_arr
    control_recv   = Signal(int, int, int)                 # steering, l_spd, r_spd
    top_img_recv   = Signal(object)                        # numpy RGB image
    task_recv      = Signal(str)                           # JSON task status
    feedback_recv  = Signal(str)                           # feedback text


# ─── ROS 노드 ─────────────────────────────────────────────────────────────
class DebugRosNode(Node):
    def __init__(self, bridge: DebugBridge):
        super().__init__("debug_gui_node")
        self._b = bridge

        rel = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         durability=QoSDurabilityPolicy.VOLATILE, depth=1)
        be  = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         durability=QoSDurabilityPolicy.VOLATILE, depth=1)

        self.create_subscription(Odometry,        "/odom",                    self._odom_cb,      be)
        self.create_subscription(LaserScan,       "lidar_processed",          self._lidar_cb,     be)
        self.create_subscription(Bool,            "lidar_obstacle_info",      self._obs_cb,       rel)
        self.create_subscription(DetectionArray,  "detections",               self._det_cb,       rel)
        self.create_subscription(LaneInfo,        "yolov8_lane_info",         self._lane_cb,      rel)
        self.create_subscription(String,          "yolov8_traffic_light_info",self._traffic_cb,   rel)
        self.create_subscription(PathPlanningResult,"path_planning_result",   self._path_cb,      rel)
        self.create_subscription(MotionCommand,   "topic_control_signal",     self._ctrl_cb,      rel)
        self.create_subscription(Image,           "/top_camera/image_raw",    self._top_img_cb,   be)
        self.create_subscription(String,          "/mission/task_status",     self._task_cb,      rel)
        self.create_subscription(String,          "/mission/feedback",        self._fb_cb,        rel)

        self._pub_cmd = self.create_publisher(String, "/mission/command", rel)
        self._pub_vla = self.create_publisher(String, "vla/goal_cmd",    rel)

    def send_command(self, text: str):
        self._pub_cmd.publish(String(data=text))

    # ── 콜백 ──────────────────────────────────────────────────────────────
    def _odom_cb(self, m: Odometry):
        q = m.pose.pose.orientation
        siny = 2*(q.w*q.z + q.x*q.y); cosy = 1 - 2*(q.y*q.y + q.z*q.z)
        self._b.odom_recv.emit(m.pose.pose.position.x,
                               m.pose.pose.position.y,
                               math.degrees(math.atan2(siny, cosy)))

    def _lidar_cb(self, m: LaserScan):
        self._b.lidar_recv.emit(np.array(m.ranges, dtype=np.float32),
                                m.angle_min, m.angle_increment)

    def _obs_cb(self, m: Bool):
        self._b.obstacle_recv.emit(m.data, 0.0)

    def _det_cb(self, m: DetectionArray):
        self._b.detection_recv.emit(m)

    def _lane_cb(self, m: LaneInfo):
        pts = [(tp.target_x, tp.target_y) for tp in m.target_points]
        self._b.lane_recv.emit(m.slope, pts)

    def _traffic_cb(self, m: String):
        self._b.traffic_recv.emit(m.data)

    def _path_cb(self, m: PathPlanningResult):
        self._b.path_recv.emit(np.array(m.x_points), np.array(m.y_points))

    def _ctrl_cb(self, m: MotionCommand):
        self._b.control_recv.emit(m.steering, m.left_speed, m.right_speed)

    def _top_img_cb(self, m: Image):
        arr = np.frombuffer(bytes(m.data), dtype=np.uint8)
        arr = arr.reshape((m.height, m.width, 3))
        if m.encoding == 'bgr8':
            arr = arr[:, :, ::-1].copy()
        self._b.top_img_recv.emit(arr)

    def _task_cb(self, m: String):
        self._b.task_recv.emit(m.data)

    def _fb_cb(self, m: String):
        self._b.feedback_recv.emit(m.data)


# ─── 트랙 맵 + 오버레이 위젯 ──────────────────────────────────────────────
class DebugMapWidget(QWidget):
    """
    트랙 지도 위에 실시간 디버그 정보를 오버레이하는 위젯.
      - GT 중앙선 / 차선
      - 자차 위치·방향
      - LiDAR 포인트 클라우드 (세계 좌표 변환)
      - YOLOv8 탐지 객체
      - 경로 계획 결과
      - 궤적 이력
      - 랜드마크
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(500, 600)

        # 차량 상태
        self._car_x  = -2.55; self._car_y = -22.71; self._car_yaw = 90.0

        # LiDAR (세계 좌표 변환된 포인트)
        self._lidar_pts: list = []

        # 탐지 객체 (class_name, bbox_cx, bbox_cy)
        self._detections: list = []

        # 경로 (세계 좌표 — 카메라 이미지 픽셀 좌표가 아님에 주의)
        self._path_x: np.ndarray = np.array([])
        self._path_y: np.ndarray = np.array([])

        # 궤적 이력
        self._trajectory: deque = deque(maxlen=200)

        # 차선 타겟 포인트 (이미지 픽셀 좌표 → 표시만)
        self._lane_targets: list = []

        # 색상
        self._C = {
            "bg":       QColor(18,  22,  32),
            "track":    QColor(55,  60,  72),
            "center":   QColor(255, 220, 50),
            "gt_pt":    QColor(255, 220, 50, 180),
            "landmark": QColor(100, 180, 255),
            "car":      QColor(0,   220, 100),
            "lidar":    QColor(255, 80,  80,  160),
            "det":      QColor(255, 160, 0),
            "path":     QColor(0,   200, 255),
            "traj":     QColor(150, 255, 150, 120),
            "grid":     QColor(40,  50,  65),
            "text":     QColor(200, 210, 220),
        }

    # ── 업데이트 ────────────────────────────────────────────────────────
    def update_vehicle(self, x, y, yaw_deg):
        self._car_x = x; self._car_y = y; self._car_yaw = yaw_deg
        self._trajectory.append((x, y))
        self.update()

    def update_lidar(self, ranges: np.ndarray, angle_min: float, angle_inc: float):
        """LaserScan → 세계 좌표 변환."""
        pts = []
        yaw_rad = math.radians(self._car_yaw)
        for i, r in enumerate(ranges):
            if not (0.8 < r < 12.0) or not math.isfinite(r):
                continue
            ang = angle_min + i * angle_inc
            lx = r * math.cos(ang)
            ly = r * math.sin(ang)
            wx = self._car_x + lx * math.cos(yaw_rad) - ly * math.sin(yaw_rad)
            wy = self._car_y + lx * math.sin(yaw_rad) + ly * math.cos(yaw_rad)
            pts.append((wx, wy))
        self._lidar_pts = pts
        self.update()

    def update_detections(self, det_msg):
        self._detections = [
            (d.class_name,
             d.bbox.center.position.x,
             d.bbox.center.position.y,
             d.bbox.size.x,
             d.bbox.size.y,
             d.score)
            for d in det_msg.detections
        ]
        self.update()

    def update_path(self, xs: np.ndarray, ys: np.ndarray):
        self._path_x = xs; self._path_y = ys
        self.update()

    def update_lane_targets(self, pts: list):
        self._lane_targets = pts
        self.update()

    # ── 좌표 변환 ────────────────────────────────────────────────────────
    def _w2s(self, wx, wy) -> QPointF:
        w, h = self.width(), self.height()
        m = 30
        xr = TRACK_BOUNDS["x_max"] - TRACK_BOUNDS["x_min"]
        yr = TRACK_BOUNDS["y_max"] - TRACK_BOUNDS["y_min"]
        sx = m + (wx - TRACK_BOUNDS["x_min"]) / xr * (w - 2*m)
        sy = h - m - (wy - TRACK_BOUNDS["y_min"]) / yr * (h - 2*m)
        return QPointF(sx, sy)

    def _m2px(self, meters: float) -> float:
        return meters / (TRACK_BOUNDS["x_max"] - TRACK_BOUNDS["x_min"]) * (self.width() - 60)

    # ── 페인트 ────────────────────────────────────────────────────────────
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._draw_bg(p)
        self._draw_grid(p)
        self._draw_track(p)
        self._draw_trajectory(p)
        self._draw_lidar(p)
        self._draw_path(p)
        self._draw_landmarks(p)
        self._draw_vehicle(p)
        self._draw_detections_overlay(p)
        self._draw_legend(p)
        p.end()

    def _draw_bg(self, p):
        p.fillRect(self.rect(), self._C["bg"])

    def _draw_grid(self, p):
        pen = QPen(self._C["grid"], 0.5, Qt.PenStyle.DotLine)
        p.setPen(pen)
        p.setFont(QFont("monospace", 7))
        for y in range(-25, 26, 5):
            a = self._w2s(TRACK_BOUNDS["x_min"], y)
            b = self._w2s(TRACK_BOUNDS["x_max"], y)
            p.drawLine(a, b)
            p.setPen(QColor(70, 80, 100))
            p.drawText(int(a.x())+2, int(a.y())-2, f"{y}m")
            p.setPen(pen)
        for x in range(-5, 21, 5):
            a = self._w2s(x, TRACK_BOUNDS["y_min"])
            b = self._w2s(x, TRACK_BOUNDS["y_max"])
            p.drawLine(a, b)
            p.setPen(QColor(70, 80, 100))
            p.drawText(int(b.x())-10, int(b.y())+12, f"{x}")
            p.setPen(pen)

    def _draw_track(self, p):
        # 도로면 (두꺼운 회색)
        lane_px = self._m2px(LANE_WIDTH_PX_SCALE * 2)
        path = QPainterPath()
        pts = [self._w2s(x, y) for x, y in TRACK_CENTERLINE]
        path.moveTo(pts[0])
        for pt in pts[1:]: path.lineTo(pt)
        p.setPen(QPen(self._C["track"], lane_px,
                      Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                      Qt.PenJoinStyle.RoundJoin))
        p.drawPath(path)

        # GT 중앙선 (노란 점선)
        p.setPen(QPen(self._C["center"], 1.2, Qt.PenStyle.DotLine))
        for i in range(len(pts)-1):
            p.drawLine(pts[i], pts[i+1])

        # GT 포인트
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(self._C["gt_pt"]))
        for x, y in TRACK_CENTERLINE:
            c = self._w2s(x, y)
            p.drawEllipse(c, 2.5, 2.5)

    def _draw_trajectory(self, p):
        traj = list(self._trajectory)
        if len(traj) < 2:
            return
        pen = QPen(self._C["traj"], 2.0)
        p.setPen(pen)
        for i in range(len(traj)-1):
            p.drawLine(self._w2s(*traj[i]), self._w2s(*traj[i+1]))

    def _draw_lidar(self, p):
        r = max(2, self._m2px(0.15))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(self._C["lidar"]))
        for wx, wy in self._lidar_pts:
            c = self._w2s(wx, wy)
            p.drawEllipse(c, r, r)

    def _draw_path(self, p):
        if len(self._path_x) < 2:
            return
        # path_planning_result는 이미지 픽셀 좌표 — 세계좌표 아님
        # 여기서는 표시 생략 (카메라 픽셀 → 세계 변환 복잡)
        # 대신 lane target points를 표시
        if self._lane_targets:
            p.setPen(QPen(self._C["path"], 2))
            p.setFont(QFont("Arial", 7))
            for i, (tx, ty) in enumerate(self._lane_targets):
                # 이미지 픽셀 → 세계 좌표 근사 (전방 카메라 기준)
                # 전방 카메라는 차량에 탑재 → 단순 표시 생략
                pass

    def _draw_landmarks(self, p):
        p.setFont(QFont("Arial", 8, QFont.Weight.Bold))
        icons = {"출발점": "🚗", "신호등/횡단보도": "🚦",
                 "장애물구역-1": "⚠", "장애물구역-2": "⚠"}
        for name, (wx, wy) in LANDMARKS.items():
            pt = self._w2s(wx, wy)
            r  = 9
            p.setPen(QPen(self._C["landmark"], 2))
            p.setBrush(QBrush(self._C["landmark"].darker(200)))
            p.drawEllipse(pt, r, r)
            p.setPen(QPen(QColor(255, 255, 255)))
            p.drawText(QRectF(pt.x()-r, pt.y()-r, 2*r, 2*r),
                       Qt.AlignmentFlag.AlignCenter, icons.get(name, "?"))
            p.setPen(QPen(self._C["landmark"]))
            p.setFont(QFont("Arial", 7))
            p.drawText(int(pt.x())+r+2, int(pt.y())+4, name)
            p.setFont(QFont("Arial", 8, QFont.Weight.Bold))

    def _draw_vehicle(self, p):
        pt = self._w2s(self._car_x, self._car_y)
        clen = max(14, self._m2px(2.5))
        cwid = max(8,  self._m2px(1.2))

        p.save()
        p.translate(pt)
        p.rotate(-(self._car_yaw - 90.0))

        # 차체
        p.setPen(QPen(self._C["car"], 1.5))
        p.setBrush(QBrush(self._C["car"].darker(180)))
        p.drawRoundedRect(QRectF(-clen/2, -cwid/2, clen, cwid), 2, 2)

        # 진행 화살표
        tip = clen/2 + 5
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(self._C["car"]))
        p.drawPolygon(QPolygonF([
            QPointF(tip+5, 0), QPointF(tip, -3), QPointF(tip, 3)
        ]))
        p.restore()

        # 좌표 라벨
        p.setPen(QPen(self._C["car"]))
        p.setFont(QFont("monospace", 7))
        p.drawText(int(pt.x())+clen+3, int(pt.y())-3,
                   f"({self._car_x:.2f}, {self._car_y:.2f})")

    def _draw_detections_overlay(self, p):
        """탐지 객체를 트랙 맵 위에 아이콘으로 표시 (차량 앞 추정)."""
        if not self._detections:
            return
        # 전방 카메라 탐지 → 세계 좌표는 모름, 차량 주변에 표시
        p.setFont(QFont("Arial", 8))
        for i, (cls, cx, cy, sw, sh, score) in enumerate(self._detections[:8]):
            # 단순히 차량 앞 방향에 아이콘만 표시
            dist_est = max(2.0, 5.0 - i * 0.5)
            yaw_r = math.radians(self._car_yaw)
            wx = self._car_x + dist_est * math.sin(yaw_r)
            wy = self._car_y + dist_est * math.cos(yaw_r)
            pt = self._w2s(wx + i * 0.5, wy)

            color = QColor(255, 80, 80) if cls == "car" else \
                    QColor(255, 200, 0) if cls == "traffic_light" else \
                    QColor(0, 200, 200)
            p.setPen(QPen(color, 1.5, Qt.PenStyle.DashLine))
            p.setBrush(QBrush(color.darker(200)))
            p.drawEllipse(pt, 7, 7)
            p.setPen(QPen(color))
            p.drawText(int(pt.x())+9, int(pt.y())+3,
                       f"{cls[:6]}({score:.0%})")

    def _draw_legend(self, p):
        items = [
            (self._C["car"],      "자차"),
            (self._C["center"],   "GT 중앙선"),
            (self._C["lidar"],    "LiDAR 포인트"),
            (self._C["traj"],     "주행 궤적"),
            (self._C["det"],      "탐지 객체"),
            (self._C["landmark"], "랜드마크"),
        ]
        x0 = 8; y0 = self.height() - 12 - len(items) * 15
        p.setFont(QFont("Arial", 7))
        for i, (col, lbl) in enumerate(items):
            y = y0 + i * 15
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(col))
            p.drawRect(x0, y-8, 10, 9)
            p.setPen(QPen(QColor(180, 190, 200)))
            p.drawText(x0+14, y, lbl)


# ─── LiDAR 극좌표 위젯 ────────────────────────────────────────────────────
class LidarPolarWidget(QWidget):
    """LiDAR 스캔 데이터를 극좌표로 표시."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self._ranges      = np.array([])
        self._angle_min   = 0.0
        self._angle_inc   = 0.0
        self._max_range   = 12.0

    def update_scan(self, ranges, angle_min, angle_inc):
        self._ranges     = ranges
        self._angle_min  = angle_min
        self._angle_inc  = angle_inc
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2
        r_max  = min(cx, cy) - 20

        p.fillRect(self.rect(), QColor(14, 18, 28))

        # 거리 원
        p.setPen(QPen(QColor(40, 55, 75), 1, Qt.PenStyle.DotLine))
        p.setFont(QFont("monospace", 7))
        for frac in [0.25, 0.5, 0.75, 1.0]:
            r = int(r_max * frac)
            p.drawEllipse(cx-r, cy-r, 2*r, 2*r)
            dist_m = self._max_range * frac
            p.setPen(QColor(80, 100, 130))
            p.drawText(cx+r+2, cy, f"{dist_m:.1f}m")
            p.setPen(QPen(QColor(40, 55, 75), 1, Qt.PenStyle.DotLine))

        # 각도 선 (30도 간격)
        for deg in range(0, 360, 30):
            rad = math.radians(deg)
            p.drawLine(cx, cy,
                       int(cx + r_max * math.cos(rad)),
                       int(cy - r_max * math.sin(rad)))

        # LiDAR 포인트
        if len(self._ranges) == 0:
            p.setPen(QPen(QColor(100, 100, 100)))
            p.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter,
                       "lidar_processed 토픽 없음")
            p.end(); return

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(255, 60, 60, 200)))
        for i, r in enumerate(self._ranges):
            if not math.isfinite(r) or r <= 0.8 or r > self._max_range:
                continue
            ang = self._angle_min + i * self._angle_inc
            px = int(cx + (r / self._max_range) * r_max * math.cos(ang))
            py = int(cy - (r / self._max_range) * r_max * math.sin(ang))
            p.drawEllipse(px-2, py-2, 4, 4)

        # 전방 화살표
        p.setPen(QPen(QColor(0, 255, 100), 2))
        p.drawLine(cx, cy, cx, cy - r_max // 2)

        # 라벨
        p.setPen(QPen(QColor(150, 200, 255)))
        p.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        p.drawText(QRectF(0, 0, w, 20),
                   Qt.AlignmentFlag.AlignCenter, "LiDAR 스캔 (극좌표)")
        p.end()


# ─── 탑 카메라 위젯 ───────────────────────────────────────────────────────
class TopCameraWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("📷 /top_camera/image_raw 대기 중...")
        self.setStyleSheet("color: #666; font-size: 13px; background: #0d1117;")
        self.setMinimumSize(400, 300)

    def update_image(self, img: np.ndarray):
        h, w, c = img.shape
        qimg = QImage(img.data, w, h, w * c, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.setPixmap(pix.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))


# ─── 사이드 패널 (실시간 데이터 테이블) ───────────────────────────────────
class SidePanelWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(280)
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(self._make_vehicle_group())
        layout.addWidget(self._make_sensor_group())
        layout.addWidget(self._make_detect_group())
        layout.addWidget(self._make_task_group())
        layout.addWidget(self._make_feedback_group(), stretch=1)

    # ── 그룹 빌더 ──────────────────────────────────────────────────────
    def _make_vehicle_group(self):
        grp = QGroupBox("🚗 차량 상태")
        g = QGridLayout(grp); g.setSpacing(2)
        rows = [
            ("X [m]",    "_lbl_x"),
            ("Y [m]",    "_lbl_y"),
            ("Yaw [°]",  "_lbl_yaw"),
            ("조향",      "_lbl_steer"),
            ("속도 L/R", "_lbl_speed"),
        ]
        for i, (lbl, attr) in enumerate(rows):
            g.addWidget(QLabel(lbl+":"), i, 0)
            w = QLabel("—")
            w.setStyleSheet("color:#7df;font-family:monospace;")
            setattr(self, attr, w)
            g.addWidget(w, i, 1)
        return grp

    def _make_sensor_group(self):
        grp = QGroupBox("📡 센서")
        g = QGridLayout(grp); g.setSpacing(2)
        rows = [
            ("LiDAR 전방", "_lbl_lidar"),
            ("장애물",      "_lbl_obs"),
            ("신호등",      "_lbl_tl"),
            ("차선 기울기", "_lbl_slope"),
            ("차선 타겟",  "_lbl_target"),
        ]
        for i, (lbl, attr) in enumerate(rows):
            g.addWidget(QLabel(lbl+":"), i, 0)
            w = QLabel("—")
            w.setStyleSheet("color:#7df;font-family:monospace;")
            setattr(self, attr, w)
            g.addWidget(w, i, 1)
        return grp

    def _make_detect_group(self):
        grp = QGroupBox("👁 YOLOv8 탐지")
        vl = QVBoxLayout(grp)
        self._det_table = QTableWidget(0, 3)
        self._det_table.setHorizontalHeaderLabels(["클래스", "신뢰도", "위치"])
        self._det_table.setMaximumHeight(110)
        self._det_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._det_table.setStyleSheet(
            "QTableWidget{background:#0d1117;color:#cdd;font-size:10px;}"
            "QHeaderView::section{background:#161b22;color:#7df;}")
        vl.addWidget(self._det_table)
        return grp

    def _make_task_group(self):
        grp = QGroupBox("📋 미션 태스크")
        g = QGridLayout(grp); g.setSpacing(2)
        rows = [("현재 태스크", "_lbl_cur_task"),
                ("큐 개수",    "_lbl_queue"),
                ("완료",       "_lbl_done"),
                ("실패",       "_lbl_fail")]
        for i, (lbl, attr) in enumerate(rows):
            g.addWidget(QLabel(lbl+":"), i, 0)
            w = QLabel("—")
            w.setStyleSheet("color:#7df;font-family:monospace;")
            setattr(self, attr, w)
            g.addWidget(w, i, 1)
        return grp

    def _make_feedback_group(self):
        grp = QGroupBox("💬 피드백")
        vl = QVBoxLayout(grp)
        self._lbl_feedback = QLabel("—")
        self._lbl_feedback.setWordWrap(True)
        self._lbl_feedback.setStyleSheet("color:#aef;font-size:10px;")
        vl.addWidget(self._lbl_feedback)
        return grp

    # ── 업데이트 메서드 ─────────────────────────────────────────────────
    def set_vehicle(self, x, y, yaw, steer, lspd, rspd):
        self._lbl_x.setText(f"{x:.3f}")
        self._lbl_y.setText(f"{y:.3f}")
        self._lbl_yaw.setText(f"{yaw:.1f}°")
        self._lbl_steer.setText(str(steer))
        self._lbl_speed.setText(f"{lspd} / {rspd}")

    def set_lidar(self, min_dist, detected):
        d_str = f"{min_dist:.2f}m" if min_dist < 50 else "—"
        self._lbl_lidar.setText(d_str)
        if detected:
            self._lbl_obs.setText("⚠️ 감지됨")
            self._lbl_obs.setStyleSheet("color:#f66;font-family:monospace;")
        else:
            self._lbl_obs.setText("✅ 없음")
            self._lbl_obs.setStyleSheet("color:#6f6;font-family:monospace;")

    def set_traffic(self, color: str):
        colors = {"Red": "#f44", "Green": "#4f4", "Yellow": "#ff4", "None": "#888"}
        self._lbl_tl.setText(color)
        self._lbl_tl.setStyleSheet(
            f"color:{colors.get(color,'#aaa')};font-family:monospace;font-weight:bold;")

    def set_lane(self, slope, n_targets):
        self._lbl_slope.setText(f"{slope:.4f}")
        self._lbl_target.setText(f"{n_targets}개")

    def set_detections(self, dets: list):
        self._det_table.setRowCount(len(dets))
        for i, (cls, cx, cy, sw, sh, score) in enumerate(dets):
            self._det_table.setItem(i, 0, QTableWidgetItem(cls))
            self._det_table.setItem(i, 1, QTableWidgetItem(f"{score:.0%}"))
            self._det_table.setItem(i, 2,
                QTableWidgetItem(f"({int(cx)},{int(cy)})"))

    def set_task(self, json_str: str):
        try:
            s = json.loads(json_str)
            cur = s.get("current")
            self._lbl_cur_task.setText(
                cur["task_type"] if cur else "없음"
            )
            self._lbl_queue.setText(str(len(s.get("queue", []))))
            self._lbl_done.setText(str(len(s.get("completed", []))))
            self._lbl_fail.setText(str(len(s.get("failed", []))))
        except Exception:
            pass

    def set_feedback(self, text: str):
        self._lbl_feedback.setText(text[:120])


# ─── 메인 윈도우 ─────────────────────────────────────────────────────────
class DebugGUIWindow(QMainWindow):
    def __init__(self, ros_node: DebugRosNode, bridge: DebugBridge):
        super().__init__()
        self._node  = ros_node
        self._b     = bridge

        # 내부 상태 캐시
        self._car_x = -2.55; self._car_y = -22.71; self._car_yaw = 90.0
        self._lidar_min = 999.0
        self._steer = 0; self._lspd = 0; self._rspd = 0

        self.setWindowTitle("🔍 자율주행 트랙 디버거")
        self.resize(1200, 780)
        self._apply_theme()
        self._build_ui()
        self._connect_signals()

    # ── UI 구성 ─────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setSpacing(4)
        root_layout.setContentsMargins(6, 6, 6, 6)

        # ── 상단 명령 입력 바 ──
        cmd_bar = QHBoxLayout()
        lbl = QLabel("명령:")
        lbl.setFixedWidth(36)
        self._cmd_input = QLineEdit()
        self._cmd_input.setPlaceholderText(
            "예: '제자리에서 한 바퀴 돌고 횡단보도로 가줘'  |  '멈춰'  |  '출발'")
        self._cmd_input.setMinimumHeight(34)
        self._cmd_input.returnPressed.connect(self._send_cmd)
        send_btn = QPushButton("전송")
        send_btn.setFixedWidth(60)
        send_btn.setMinimumHeight(34)
        send_btn.setStyleSheet(
            "QPushButton{background:#1a5a9a;color:white;border-radius:4px;}"
            "QPushButton:hover{background:#2a6aaa;}")
        send_btn.clicked.connect(self._send_cmd)
        cmd_bar.addWidget(lbl)
        cmd_bar.addWidget(self._cmd_input)
        cmd_bar.addWidget(send_btn)
        root_layout.addLayout(cmd_bar)

        # ── 메인 영역: 사이드 패널 + 탭 뷰 ──
        body = QHBoxLayout()
        body.setSpacing(6)

        # 왼쪽: 실시간 데이터 패널
        self._side = SidePanelWidget()
        body.addWidget(self._side)

        # 오른쪽: 탭 (트랙맵 / 탑카메라 / LiDAR 극좌표)
        tabs = QTabWidget()
        tabs.setStyleSheet(
            "QTabBar::tab{min-width:120px;padding:6px 12px;}"
            "QTabBar::tab:selected{background:#1e2a40;color:#fff;}"
        )

        self._map_widget   = DebugMapWidget()
        self._top_cam      = TopCameraWidget()
        self._lidar_polar  = LidarPolarWidget()

        tabs.addTab(self._map_widget,  "🗺 트랙 맵")
        tabs.addTab(self._top_cam,     "📷 탑 카메라")
        tabs.addTab(self._lidar_polar, "📡 LiDAR 극좌표")

        body.addWidget(tabs, stretch=1)
        root_layout.addLayout(body, stretch=1)

        # ── 상태 바 ──
        self._status_bar = self.statusBar()
        self._status_bar.showMessage("ROS 연결 대기 중...")

    # ── 시그널 연결 ─────────────────────────────────────────────────────
    def _connect_signals(self):
        b = self._b
        b.odom_recv.connect(self._on_odom)
        b.lidar_recv.connect(self._on_lidar)
        b.obstacle_recv.connect(self._on_obstacle)
        b.detection_recv.connect(self._on_detection)
        b.lane_recv.connect(self._on_lane)
        b.traffic_recv.connect(self._on_traffic)
        b.path_recv.connect(self._on_path)
        b.control_recv.connect(self._on_control)
        b.top_img_recv.connect(self._on_top_image)
        b.task_recv.connect(self._on_task)
        b.feedback_recv.connect(self._on_feedback)

    # ── 슬롯 ───────────────────────────────────────────────────────────
    @Slot(float, float, float)
    def _on_odom(self, x, y, yaw):
        self._car_x = x; self._car_y = y; self._car_yaw = yaw
        self._map_widget.update_vehicle(x, y, yaw)
        self._side.set_vehicle(x, y, yaw,
                               self._steer, self._lspd, self._rspd)
        self._status_bar.showMessage(
            f"위치: ({x:.2f}, {y:.2f})  방향: {yaw:.1f}°  "
            f"조향: {self._steer}  속도: {self._lspd}")

    @Slot(object, float, float)
    def _on_lidar(self, ranges, angle_min, angle_inc):
        self._map_widget.update_lidar(ranges, angle_min, angle_inc)
        self._lidar_polar.update_scan(ranges, angle_min, angle_inc)
        valid = ranges[(np.isfinite(ranges)) & (ranges > 0.8) & (ranges < 12)]
        self._lidar_min = float(np.min(valid)) if len(valid) > 0 else 999.0
        self._side.set_lidar(self._lidar_min, self._lidar_min < 3.0)

    @Slot(bool, float)
    def _on_obstacle(self, detected, _):
        self._side.set_lidar(self._lidar_min, detected)

    @Slot(object)
    def _on_detection(self, msg):
        dets = [(d.class_name, d.bbox.center.position.x,
                 d.bbox.center.position.y, d.bbox.size.x,
                 d.bbox.size.y, d.score)
                for d in msg.detections]
        self._map_widget.update_detections(msg)
        self._side.set_detections(dets)

    @Slot(float, object)
    def _on_lane(self, slope, targets):
        self._map_widget.update_lane_targets(targets)
        self._side.set_lane(slope, len(targets))

    @Slot(str)
    def _on_traffic(self, color):
        self._side.set_traffic(color)

    @Slot(object, object)
    def _on_path(self, xs, ys):
        self._map_widget.update_path(xs, ys)

    @Slot(int, int, int)
    def _on_control(self, steer, lspd, rspd):
        self._steer = steer; self._lspd = lspd; self._rspd = rspd
        self._side.set_vehicle(self._car_x, self._car_y, self._car_yaw,
                               steer, lspd, rspd)

    @Slot(object)
    def _on_top_image(self, img):
        self._top_cam.update_image(img)

    @Slot(str)
    def _on_task(self, json_str):
        self._side.set_task(json_str)

    @Slot(str)
    def _on_feedback(self, text):
        self._side.set_feedback(text)

    def _send_cmd(self):
        cmd = self._cmd_input.text().strip()
        if cmd:
            self._node.send_command(cmd)
            self._status_bar.showMessage(f"명령 전송: {cmd[:80]}")
            self._cmd_input.clear()

    # ── 다크 테마 ────────────────────────────────────────────────────────
    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow,QWidget{background:#13171f;color:#c9d1d9;}
            QGroupBox{border:1px solid #2a3a4a;border-radius:4px;
                      margin-top:8px;padding-top:4px;
                      font-weight:bold;color:#8ab4d4;}
            QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px;}
            QLineEdit{background:#1a1e2a;border:1px solid #2a3a5a;
                      border-radius:4px;padding:4px 8px;color:#c9d1d9;}
            QLineEdit:focus{border-color:#3a6496;}
            QPushButton{background:#1e2a3a;border:1px solid #2a4a6a;
                        border-radius:4px;padding:4px 10px;color:#aacce8;}
            QPushButton:hover{background:#253a4a;}
            QTabWidget::pane{border:1px solid #2a3a4a;}
            QTabBar::tab{background:#1a2030;color:#8ab4d4;
                         padding:6px 14px;border:1px solid #2a3a4a;}
            QTabBar::tab:selected{background:#1e2a40;color:#ffffff;}
            QStatusBar{color:#8ab4d4;}
            QLabel{color:#aabbcc;}
        """)


# ─── 진입점 ──────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    bridge   = DebugBridge()
    ros_node = DebugRosNode(bridge)

    spin_thread = threading.Thread(
        target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()

    app    = QApplication(sys.argv)
    window = DebugGUIWindow(ros_node, bridge)
    window.show()

    code = app.exec()
    ros_node.destroy_node()
    rclpy.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()

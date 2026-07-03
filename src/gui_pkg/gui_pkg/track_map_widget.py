"""
track_map_widget.py
===================
트랙 지도 시각화 위젯 (PySide6)

기능:
  - 트랙 중앙선 / 차선 경계 표시
  - 자차 실시간 위치·방향 표시
  - 랜드마크 (신호등, 장애물 구역 등) 표시
  - 태스크 목표 하이라이트
  - 장애물 위치 표시
  - GT 웨이포인트 클릭 시 목적지 선택
  - 트랙 전체 정보 패널
"""

import math
from typing import Optional, List, Tuple

from PySide6.QtWidgets import QWidget, QToolTip
from PySide6.QtCore import Qt, QPointF, Signal, QRectF
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPainterPath,
    QLinearGradient, QMouseEvent,
)

# ─── 트랙 GT 데이터 (인라인 — GUI 단독 실행 지원) ─────────────────────
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
    "출발점":       (-2.55, -22.71, "🚗"),
    "신호등/횡단보도": (-5.63,  17.90, "🚦"),
    "장애물구역-1":  (-3.66,   8.71, "⚠️"),
    "장애물구역-2":  (-3.66,   2.04, "⚠️"),
    "남쪽커브":     ( 7.00, -23.00, "↩"),
    "북쪽커브":     ( 5.00,  21.00, "↪"),
}

TRACK_BOUNDS = {
    "x_min": -8.0, "x_max": 20.0,
    "y_min": -26.0, "y_max": 25.0,
}

LANE_WIDTH = 2.8   # 차선 폭 [m]


class TrackMapWidget(QWidget):
    """
    트랙 지도 위젯

    시그널:
      landmark_clicked(str)  - 랜드마크 클릭 시 목적지 이름 반환
    """

    landmark_clicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 560)
        self.setMouseTracking(True)

        # ── 자차 상태 ──
        self._car_x:   float = -2.55
        self._car_y:   float = -22.71
        self._car_yaw: float = 90.0   # degrees

        # ── 장애물 ──
        self._obstacles: List[Tuple[float, float, float]] = []  # (x, y, dist)

        # ── 태스크 목표 ──
        self._target:      Optional[str] = None
        self._target_pos:  Optional[Tuple[float, float]] = None

        # ── 차선 변경 표시 ──
        self._in_lane_change: bool = False
        self._lc_state:       str  = "IDLE"

        # ── 클릭 가능 영역 ──
        self._landmark_rects: dict = {}  # name → QRectF (화면 좌표)

        # ── 색상 팔레트 ──
        self._bg_color         = QColor(20, 25, 35)
        self._track_color      = QColor(60, 65, 75)
        self._center_color     = QColor(255, 220, 50)
        self._lane_mark_color  = QColor(255, 255, 255, 160)
        self._car_color        = QColor(0, 200, 100)
        self._obstacle_color   = QColor(255, 60, 60)
        self._landmark_color   = QColor(100, 180, 255)
        self._target_color     = QColor(255, 120, 0)
        self._grid_color       = QColor(40, 50, 65)

    # ─── 상태 업데이트 (외부에서 호출) ──────────────────────────────────
    def update_vehicle(self, x: float, y: float, yaw_deg: float):
        self._car_x   = x
        self._car_y   = y
        self._car_yaw = yaw_deg
        self.update()

    def update_obstacles(self, obstacles: List[Tuple[float, float, float]]):
        """장애물 목록 갱신. obstacles: [(x, y, dist), ...]"""
        self._obstacles = obstacles
        self.update()

    def update_target(self, target_name: Optional[str]):
        """현재 목표 랜드마크 갱신."""
        self._target = target_name
        landmark_map = {
            "crosswalk":     (-5.63, 17.90),
            "traffic_light": (-5.63, 17.90),
            "start":         (-2.55, -22.71),
            "obstacle_1":    (-3.66,  8.71),
            "obstacle_2":    (-3.66,  2.04),
            "mid":           (-2.55, -2.00),
        }
        self._target_pos = landmark_map.get(target_name)
        self.update()

    def update_lane_change(self, in_lc: bool, lc_state: str):
        self._in_lane_change = in_lc
        self._lc_state       = lc_state
        self.update()

    # ─── 좌표 변환 ────────────────────────────────────────────────────
    def _world_to_screen(self, wx: float, wy: float) -> QPointF:
        """Gazebo 세계 좌표 → 위젯 화면 좌표."""
        w, h = self.width(), self.height()
        margin = 30

        x_range = TRACK_BOUNDS["x_max"] - TRACK_BOUNDS["x_min"]
        y_range = TRACK_BOUNDS["y_max"] - TRACK_BOUNDS["y_min"]

        sx = margin + (wx - TRACK_BOUNDS["x_min"]) / x_range * (w - 2 * margin)
        # Y축 반전 (화면은 위가 작은 값)
        sy = h - margin - (wy - TRACK_BOUNDS["y_min"]) / y_range * (h - 2 * margin)
        return QPointF(sx, sy)

    def _screen_to_world(self, sx: float, sy: float) -> Tuple[float, float]:
        """화면 좌표 → Gazebo 세계 좌표."""
        w, h = self.width(), self.height()
        margin = 30
        x_range = TRACK_BOUNDS["x_max"] - TRACK_BOUNDS["x_min"]
        y_range = TRACK_BOUNDS["y_max"] - TRACK_BOUNDS["y_min"]
        wx = (sx - margin) / (w - 2 * margin) * x_range + TRACK_BOUNDS["x_min"]
        wy = TRACK_BOUNDS["y_min"] + (1 - (sy - margin) / (h - 2 * margin)) * y_range
        return wx, wy

    def _m_to_px(self, meters: float) -> float:
        """미터 → 픽셀 스케일."""
        w = self.width() - 60
        x_range = TRACK_BOUNDS["x_max"] - TRACK_BOUNDS["x_min"]
        return meters / x_range * w

    # ─── 페인트 이벤트 ────────────────────────────────────────────────
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        self._draw_background(painter)
        self._draw_grid(painter)
        self._draw_track(painter)
        self._draw_centerline(painter)
        self._draw_landmarks(painter)
        self._draw_target(painter)
        self._draw_obstacles(painter)
        self._draw_vehicle(painter)
        self._draw_legend(painter)
        self._draw_info_overlay(painter)

        painter.end()

    def _draw_background(self, p: QPainter):
        p.fillRect(self.rect(), self._bg_color)

    def _draw_grid(self, p: QPainter):
        pen = QPen(self._grid_color, 0.5, Qt.PenStyle.DotLine)
        p.setPen(pen)
        font = QFont("monospace", 7)
        p.setFont(font)

        # Y축 그리드 (-25 ~ 25, 5단위)
        for y_val in range(-25, 26, 5):
            pt1 = self._world_to_screen(TRACK_BOUNDS["x_min"], y_val)
            pt2 = self._world_to_screen(TRACK_BOUNDS["x_max"], y_val)
            p.drawLine(pt1, pt2)
            p.setPen(QColor(80, 90, 110))
            p.drawText(int(pt1.x()) + 2, int(pt1.y()) - 2, f"{y_val}m")
            p.setPen(pen)

        # X축 그리드 (-5 ~ 20, 5단위)
        for x_val in range(-5, 21, 5):
            pt1 = self._world_to_screen(x_val, TRACK_BOUNDS["y_min"])
            pt2 = self._world_to_screen(x_val, TRACK_BOUNDS["y_max"])
            p.drawLine(pt1, pt2)

    def _draw_track(self, p: QPainter):
        """트랙 도로면 그리기 (두꺼운 회색 라인)."""
        lane_px = self._m_to_px(LANE_WIDTH * 2.0)  # 양방향 도로 폭
        path = QPainterPath()

        pts = [self._world_to_screen(x, y) for x, y in TRACK_CENTERLINE]
        if pts:
            path.moveTo(pts[0])
            for pt in pts[1:]:
                path.lineTo(pt)

        pen = QPen(self._track_color, lane_px, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.drawPath(path)

        # 차선 마킹 (흰색 점선)
        pen2 = QPen(self._lane_mark_color, 1.5, Qt.PenStyle.DashLine)
        p.setPen(pen2)
        p.drawPath(path)

    def _draw_centerline(self, p: QPainter):
        """트랙 중앙선 그리기 (노란 점선)."""
        pen = QPen(self._center_color, 1.0, Qt.PenStyle.DotLine)
        p.setPen(pen)

        pts = [self._world_to_screen(x, y) for x, y in TRACK_CENTERLINE]
        for i in range(len(pts) - 1):
            p.drawLine(pts[i], pts[i + 1])

        # GT 포인트 표시
        p.setPen(QPen(self._center_color, 1))
        p.setBrush(QBrush(self._center_color))
        for x, y in TRACK_CENTERLINE:
            pt = self._world_to_screen(x, y)
            r = 2.5
            p.drawEllipse(pt, r, r)

    def _draw_landmarks(self, p: QPainter):
        """랜드마크 표시."""
        self._landmark_rects.clear()
        font = QFont("Arial", 9, QFont.Weight.Bold)
        p.setFont(font)

        for name, (wx, wy, icon) in LANDMARKS.items():
            pt = self._world_to_screen(wx, wy)
            r  = 10

            # 원형 배경
            is_target = (self._target and
                         (("신호등" in name and self._target in ["crosswalk", "traffic_light"])
                          or ("출발점" in name and self._target == "start")))
            color = self._target_color if is_target else self._landmark_color
            p.setPen(QPen(color, 2))
            p.setBrush(QBrush(color.darker(180)))
            p.drawEllipse(pt, r, r)

            # 아이콘 텍스트
            p.setPen(QPen(QColor(255, 255, 255)))
            p.drawText(QRectF(pt.x() - r, pt.y() - r, 2*r, 2*r),
                       Qt.AlignmentFlag.AlignCenter, icon[:1])

            # 라벨
            p.setPen(QPen(color))
            label_font = QFont("Arial", 7)
            p.setFont(label_font)
            p.drawText(int(pt.x()) + r + 3, int(pt.y()) + 4, name)
            p.setFont(font)

            # 클릭 가능 영역 저장
            rect = QRectF(pt.x() - r - 2, pt.y() - r - 2, 2*r + 4 + 80, 2*r + 4)
            self._landmark_rects[name] = rect

    def _draw_target(self, p: QPainter):
        """현재 목표 하이라이트."""
        if self._target_pos is None:
            return
        wx, wy = self._target_pos
        pt = self._world_to_screen(wx, wy)

        # 펄스 링
        pen = QPen(self._target_color, 2, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for r in [18, 26, 34]:
            p.drawEllipse(pt, r, r)

        # 중앙 점
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(self._target_color))
        p.drawEllipse(pt, 5, 5)

        # "목표" 라벨
        p.setPen(QPen(self._target_color))
        p.setFont(QFont("Arial", 8, QFont.Weight.Bold))
        p.drawText(int(pt.x()) + 8, int(pt.y()) - 8, f"목표: {self._target or ''}")

    def _draw_obstacles(self, p: QPainter):
        """장애물 위치 표시."""
        for ox, oy, dist in self._obstacles:
            pt = self._world_to_screen(ox, oy)
            r  = max(6, self._m_to_px(0.8))

            # 빨간 원
            p.setPen(QPen(self._obstacle_color, 2))
            p.setBrush(QBrush(self._obstacle_color.darker(160)))
            p.drawEllipse(pt, r, r)

            # 거리 라벨
            p.setPen(QPen(self._obstacle_color))
            p.setFont(QFont("Arial", 7))
            p.drawText(int(pt.x()) + r + 2, int(pt.y()) + 3, f"{dist:.1f}m")

    def _draw_vehicle(self, p: QPainter):
        """자차 위치·방향 표시."""
        pt = self._world_to_screen(self._car_x, self._car_y)
        car_len_px = max(14, self._m_to_px(2.5))
        car_wid_px = max(8,  self._m_to_px(1.2))

        p.save()
        p.translate(pt)
        # Gazebo yaw: 0=+X, π/2=+Y → 화면 각도 조정
        # 화면 Y가 반전되므로 각도 부호 반전
        screen_angle = -(self._car_yaw - 90.0)
        p.rotate(screen_angle)

        # 차량 바디
        color = self._car_color if not self._in_lane_change else QColor(255, 200, 0)
        p.setPen(QPen(color, 1.5))
        p.setBrush(QBrush(color.darker(170)))
        rect = QRectF(-car_len_px / 2, -car_wid_px / 2, car_len_px, car_wid_px)
        p.drawRoundedRect(rect, 2, 2)

        # 진행 방향 화살표
        p.setPen(QPen(color, 2))
        p.setBrush(QBrush(color))
        tip_x = car_len_px / 2 + 4
        p.drawPolygon([
            QPointF(tip_x + 5, 0),
            QPointF(tip_x, -3),
            QPointF(tip_x, 3),
        ])

        p.restore()

        # 위치 라벨
        p.setPen(QPen(self._car_color))
        p.setFont(QFont("monospace", 7))
        p.drawText(int(pt.x()) + car_len_px + 3, int(pt.y()) - 3,
                   f"({self._car_x:.1f}, {self._car_y:.1f})")

    def _draw_legend(self, p: QPainter):
        """범례 표시."""
        items = [
            (self._car_color,      "자차"),
            (self._center_color,   "GT 중앙선"),
            (self._landmark_color, "랜드마크"),
            (self._target_color,   "목표"),
            (self._obstacle_color, "장애물"),
        ]
        x0, y0 = 8, self.height() - 12 - len(items) * 16
        p.setFont(QFont("Arial", 8))
        for i, (color, label) in enumerate(items):
            y = y0 + i * 16
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(color))
            p.drawRect(x0, y - 8, 12, 10)
            p.setPen(QPen(QColor(200, 200, 200)))
            p.drawText(x0 + 16, y, label)

    def _draw_info_overlay(self, p: QPainter):
        """차선 변경 상태 표시."""
        if self._in_lane_change:
            p.setPen(QPen(QColor(255, 200, 0)))
            p.setFont(QFont("Arial", 11, QFont.Weight.Bold))
            p.drawText(QRectF(0, 0, self.width(), 30),
                       Qt.AlignmentFlag.AlignCenter,
                       f"🔄 차선 변경 중 [{self._lc_state}]")

    # ─── 마우스 이벤트 ────────────────────────────────────────────────
    def mousePressEvent(self, event: QMouseEvent):
        pos = event.position()
        for name, rect in self._landmark_rects.items():
            if rect.contains(pos):
                # 목적지 이름 매핑
                target_map = {
                    "출발점":          "start",
                    "신호등/횡단보도":  "crosswalk",
                    "장애물구역-1":    "obstacle_1",
                    "장애물구역-2":    "obstacle_2",
                }
                target_key = target_map.get(name, name)
                self.landmark_clicked.emit(target_key)
                return

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
        for name, rect in self._landmark_rects.items():
            if rect.contains(pos):
                wx, wy = self._screen_to_world(pos.x(), pos.y())
                QToolTip.showText(
                    event.globalPosition().toPoint(),
                    f"{name}\n세계좌표: ({wx:.1f}, {wy:.1f})",
                    self
                )
                return
        QToolTip.hideText()

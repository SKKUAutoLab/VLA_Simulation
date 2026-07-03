#!/usr/bin/env python3
"""
mission_gui_node.py
===================
미션 제어 PySide6 GUI 노드

기능:
  - 자연어 명령 입력 및 전송
  - 태스크 큐 실시간 표시 (pending / in_progress / completed / failed)
  - 트랙 맵 시각화 (자차 위치, 장애물, GT 웨이포인트, 목표 하이라이트)
  - 미션 시뮬레이션 설정 (장애물 리스폰, 신호등 상태)
  - 실시간 로그 패널
  - 미리 정의된 빠른 명령 버튼

실행:
  ros2 run gui_pkg mission_gui_node
"""

import sys
import json
import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSHistoryPolicy,
                        QoSDurabilityPolicy, QoSReliabilityPolicy)
from std_msgs.msg import String, Bool
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QListWidget, QListWidgetItem,
    QTextEdit, QGroupBox, QSplitter, QStatusBar, QFrame,
    QGridLayout, QScrollArea, QComboBox, QCheckBox, QSlider,
    QTabWidget, QTableWidget, QTableWidgetItem,
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QThread, Slot
from PySide6.QtGui import QColor, QFont, QPalette, QTextCursor

import numpy as np

from .track_map_widget import TrackMapWidget

# ─── ROS 브리지 (스레드 안전 시그널 전달) ─────────────────────────────
class RosBridge(QObject):
    """ROS 콜백 → Qt 시그널 변환 (스레드 안전)."""

    status_received   = Signal(str)   # task_status JSON
    feedback_received = Signal(str)   # 실시간 피드백
    reasoning_received= Signal(str)   # 실패 이유
    vla_status_recv   = Signal(str)   # VLA 상태
    odom_received     = Signal(float, float, float)  # x, y, yaw_deg
    obstacle_received = Signal(bool, float)           # detected, min_dist

    def __init__(self):
        super().__init__()


# ─── ROS 노드 ────────────────────────────────────────────────────────────
class GUINode(Node):
    """GUI용 ROS 노드 (백그라운드 스레드에서 spin)."""

    def __init__(self, bridge: RosBridge):
        super().__init__("mission_gui_node")
        self._bridge = bridge

        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE, depth=1,
        )
        best_effort_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE, depth=1,
        )

        # 구독자
        self.create_subscription(String, "/mission/task_status",
                                 self._status_cb,   reliable_qos)
        self.create_subscription(String, "/mission/feedback",
                                 self._feedback_cb, reliable_qos)
        self.create_subscription(String, "/mission/reasoning",
                                 self._reason_cb,   reliable_qos)
        self.create_subscription(String, "vla/status",
                                 self._vla_cb,      reliable_qos)
        self.create_subscription(Odometry, "/odom",
                                 self._odom_cb,     best_effort_qos)
        self.create_subscription(Bool, "lidar_obstacle_info",
                                 self._obstacle_cb, reliable_qos)

        # 퍼블리셔
        self._pub_cmd    = self.create_publisher(String, "/mission/command", reliable_qos)
        self._pub_vla    = self.create_publisher(String, "vla/goal_cmd",    reliable_qos)

        self.get_logger().info("GUINode 시작")

    def send_mission_command(self, cmd: str):
        msg = String()
        msg.data = cmd
        self._pub_cmd.publish(msg)

    def send_vla_command(self, cmd: str):
        msg = String()
        msg.data = cmd
        self._pub_vla.publish(msg)

    def _status_cb(self, msg: String):
        self._bridge.status_received.emit(msg.data)

    def _feedback_cb(self, msg: String):
        self._bridge.feedback_received.emit(msg.data)

    def _reason_cb(self, msg: String):
        self._bridge.reasoning_received.emit(msg.data)

    def _vla_cb(self, msg: String):
        self._bridge.vla_status_recv.emit(msg.data)

    def _odom_cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw_deg = math.degrees(math.atan2(siny, cosy))
        self._bridge.odom_received.emit(
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            yaw_deg,
        )

    def _obstacle_cb(self, msg: Bool):
        self._bridge.obstacle_received.emit(msg.data, 0.0)


# ─── 빠른 명령 버튼 정의 ─────────────────────────────────────────────
QUICK_COMMANDS = [
    ("🚦 횡단보도로 가기\n(장애물 회피 포함)",
     "횡단보도까지 트랙을 따라 주행하되 중간에 만나는 장애물 차량이 있다면 차선 변경을 하여 회피해서 도착지점에 도달하게 해줘"),
    ("🔄 제자리 돌고\n횡단보도로",
     "제자리에서 한 바퀴를 돌고 횡단보도로 가줘"),
    ("🔄 360도 회전",
     "제자리에서 한 바퀴 돌아줘"),
    ("🚦 신호등 직접",
     "신호등 직접 가줘"),
    ("🏠 출발점 복귀",
     "출발점으로 돌아가"),
    ("⛔ 정지",
     "멈춰"),
    ("▶ 재개",
     "출발"),
    ("🔄 180도 + 출발점",
     "반 바퀴 돌고 출발점으로 돌아가"),
]

STATUS_COLORS = {
    "pending":     QColor(150, 150, 150),
    "in_progress": QColor(50,  200, 50),
    "completed":   QColor(100, 180, 255),
    "failed":      QColor(255, 80,  80),
}
STATUS_LABELS = {
    "pending":     "대기",
    "in_progress": "실행 중",
    "completed":   "완료",
    "failed":      "실패",
}


# ─── 메인 윈도우 ─────────────────────────────────────────────────────────
class MissionGUIWindow(QMainWindow):

    def __init__(self, ros_node: GUINode, bridge: RosBridge):
        super().__init__()
        self._node   = ros_node
        self._bridge = bridge
        self._latest_status: dict = {}

        self.setWindowTitle("🚗 자율주행 미션 제어 GUI")
        self.resize(1280, 800)
        self._apply_dark_theme()
        self._build_ui()
        self._connect_signals()

        # UI 갱신 타이머 (100ms)
        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(self._refresh_ui)
        self._ui_timer.start(200)

        self.statusBar().showMessage("ROS 연결 대기 중...")

    # ─── UI 구성 ────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # ── 왼쪽 패널 ──
        left = QWidget()
        left.setFixedWidth(420)
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(6)

        left_layout.addWidget(self._build_command_panel())
        left_layout.addWidget(self._build_quick_buttons())
        left_layout.addWidget(self._build_task_queue_panel(), stretch=1)
        left_layout.addWidget(self._build_vehicle_info_panel())

        # ── 오른쪽 패널 (탭) ──
        right_tabs = QTabWidget()
        right_tabs.setStyleSheet("QTabBar::tab { min-width: 100px; }")

        # 탭 1: 트랙 맵
        self._track_map = TrackMapWidget()
        right_tabs.addTab(self._track_map, "🗺 트랙 맵")

        # 탭 2: 로그
        log_widget = self._build_log_panel()
        right_tabs.addTab(log_widget, "📋 로그")

        # 탭 3: GT 데이터
        gt_widget = self._build_gt_panel()
        right_tabs.addTab(gt_widget, "📍 GT 데이터")

        # 탭 4: 시뮬레이션 설정
        sim_widget = self._build_sim_panel()
        right_tabs.addTab(sim_widget, "⚙️ 시뮬 설정")

        main_layout.addWidget(left)
        main_layout.addWidget(right_tabs, stretch=1)

    def _build_command_panel(self) -> QGroupBox:
        grp = QGroupBox("💬 명령 입력")
        layout = QVBoxLayout(grp)

        self._cmd_input = QLineEdit()
        self._cmd_input.setPlaceholderText(
            "예: '제자리에서 한 바퀴 돌고 횡단보도로 가줘'"
        )
        self._cmd_input.setMinimumHeight(36)
        self._cmd_input.returnPressed.connect(self._on_send_command)
        layout.addWidget(self._cmd_input)

        btn_row = QHBoxLayout()
        send_btn = QPushButton("📤 전송")
        send_btn.setMinimumHeight(32)
        send_btn.setStyleSheet(
            "QPushButton { background: #2a6496; color: white; border-radius: 4px; }"
            "QPushButton:hover { background: #3a74a6; }"
        )
        send_btn.clicked.connect(self._on_send_command)

        clear_btn = QPushButton("🗑 지우기")
        clear_btn.setMinimumHeight(32)
        clear_btn.clicked.connect(lambda: self._cmd_input.clear())

        btn_row.addWidget(send_btn)
        btn_row.addWidget(clear_btn)
        layout.addLayout(btn_row)

        return grp

    def _build_quick_buttons(self) -> QGroupBox:
        grp = QGroupBox("⚡ 빠른 명령")
        grid = QGridLayout(grp)
        grid.setSpacing(4)

        for i, (label, cmd) in enumerate(QUICK_COMMANDS):
            btn = QPushButton(label)
            btn.setMinimumHeight(48)
            btn.setStyleSheet(
                "QPushButton { background: #1e3a4a; color: #ddd; "
                "border: 1px solid #2a5a7a; border-radius: 4px; font-size: 11px; }"
                "QPushButton:hover { background: #254a5a; }"
                "QPushButton:pressed { background: #2a6496; }"
            )
            btn.clicked.connect(lambda checked, c=cmd: self._send_command(c))
            grid.addWidget(btn, i // 2, i % 2)

        return grp

    def _build_task_queue_panel(self) -> QGroupBox:
        grp = QGroupBox("📋 태스크 큐")
        layout = QVBoxLayout(grp)

        self._task_list = QListWidget()
        self._task_list.setStyleSheet(
            "QListWidget { background: #1a1e2a; border: 1px solid #2a3a4a; }"
            "QListWidget::item { padding: 4px; border-bottom: 1px solid #2a3a4a; }"
        )
        layout.addWidget(self._task_list)

        return grp

    def _build_vehicle_info_panel(self) -> QGroupBox:
        grp = QGroupBox("🚗 자차 상태")
        layout = QGridLayout(grp)
        layout.setSpacing(4)

        self._lbl_pos = QLabel("위치: (?, ?)")
        self._lbl_yaw = QLabel("방향: ?°")
        self._lbl_obs = QLabel("장애물: 없음")
        self._lbl_lcs = QLabel("차선변경: IDLE")
        self._lbl_prg = QLabel("진행률: ?")

        for i, lbl in enumerate([self._lbl_pos, self._lbl_yaw,
                                  self._lbl_obs, self._lbl_lcs, self._lbl_prg]):
            lbl.setStyleSheet("color: #aaddff; font-family: monospace;")
            layout.addWidget(lbl, i // 2, i % 2)

        return grp

    def _build_log_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setStyleSheet(
            "QTextEdit { background: #0d1117; color: #c9d1d9; "
            "font-family: 'Courier New', monospace; font-size: 11px; "
            "border: 1px solid #30363d; }"
        )
        layout.addWidget(self._log_text)

        btn_row = QHBoxLayout()
        clear_btn = QPushButton("🗑 로그 지우기")
        clear_btn.clicked.connect(self._log_text.clear)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        return w

    def _build_gt_panel(self) -> QWidget:
        """GT 데이터 테이블 표시."""
        w = QWidget()
        layout = QVBoxLayout(w)

        lbl = QLabel("트랙 GT 중앙선 웨이포인트 (Gazebo 세계 좌표)")
        lbl.setStyleSheet("color: #ffd700; font-weight: bold; margin: 4px;")
        layout.addWidget(lbl)

        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["#", "X [m]", "Y [m]", "설명"])
        table.setStyleSheet(
            "QTableWidget { background: #0d1117; color: #c9d1d9; "
            "gridline-color: #30363d; }"
            "QHeaderView::section { background: #161b22; color: #8b949e; }"
        )

        from .track_map_widget import TRACK_CENTERLINE, LANDMARKS

        # 랜드마크 역매핑
        landmark_at = {}
        for name, (lx, ly, _) in LANDMARKS.items():
            landmark_at[(round(lx, 1), round(ly, 1))] = name

        table.setRowCount(len(TRACK_CENTERLINE))
        for i, (x, y) in enumerate(TRACK_CENTERLINE):
            key = (round(x, 1), round(y, 1))
            desc = landmark_at.get(key, "")
            table.setItem(i, 0, QTableWidgetItem(str(i)))
            table.setItem(i, 1, QTableWidgetItem(f"{x:.3f}"))
            table.setItem(i, 2, QTableWidgetItem(f"{y:.3f}"))
            table.setItem(i, 3, QTableWidgetItem(desc))
            if desc:
                for col in range(4):
                    item = table.item(i, col)
                    if item:
                        item.setBackground(QColor(40, 60, 40))

        table.resizeColumnsToContents()
        layout.addWidget(table)

        # 랜드마크 테이블
        lbl2 = QLabel("주요 랜드마크")
        lbl2.setStyleSheet("color: #ffd700; font-weight: bold; margin: 4px;")
        layout.addWidget(lbl2)

        lm_table = QTableWidget()
        lm_table.setColumnCount(3)
        lm_table.setHorizontalHeaderLabels(["이름", "X [m]", "Y [m]"])
        lm_table.setStyleSheet(table.styleSheet())
        lm_table.setMaximumHeight(180)
        lm_table.setRowCount(len(LANDMARKS))
        for i, (name, (x, y, _)) in enumerate(LANDMARKS.items()):
            lm_table.setItem(i, 0, QTableWidgetItem(name))
            lm_table.setItem(i, 1, QTableWidgetItem(f"{x:.3f}"))
            lm_table.setItem(i, 2, QTableWidgetItem(f"{y:.3f}"))
        lm_table.resizeColumnsToContents()
        layout.addWidget(lm_table)

        return w

    def _build_sim_panel(self) -> QWidget:
        """시뮬레이션 설정 패널."""
        w = QWidget()
        layout = QVBoxLayout(w)

        # 장애물 차량 제어
        obs_grp = QGroupBox("🚧 장애물 차량")
        obs_layout = QGridLayout(obs_grp)

        obs_layout.addWidget(QLabel("장애물 1 위치:"), 0, 0)
        self._obs1_combo = QComboBox()
        self._obs1_combo.addItems([
            "장애물구역-1 (y=8.71)",
            "장애물구역-2 (y=2.04)",
            "비활성",
        ])
        obs_layout.addWidget(self._obs1_combo, 0, 1)

        obs_layout.addWidget(QLabel("장애물 2 위치:"), 1, 0)
        self._obs2_combo = QComboBox()
        self._obs2_combo.addItems([
            "장애물구역-2 (y=2.04)",
            "장애물구역-1 (y=8.71)",
            "비활성",
        ])
        obs_layout.addWidget(self._obs2_combo, 1, 1)

        respawn_btn = QPushButton("🔄 장애물 리스폰")
        respawn_btn.clicked.connect(self._on_respawn_obstacles)
        obs_layout.addWidget(respawn_btn, 2, 0, 1, 2)

        layout.addWidget(obs_grp)

        # 신호등
        tl_grp = QGroupBox("🚦 신호등")
        tl_layout = QVBoxLayout(tl_grp)
        tl_btn_row = QHBoxLayout()
        for color, label in [("red", "🔴 적색"), ("green", "🟢 녹색"), ("yellow", "🟡 황색")]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, c=color: self._log(f"신호등 설정: {c} (시뮬레이션에서 직접 변경 필요)"))
            tl_btn_row.addWidget(btn)
        tl_layout.addLayout(tl_btn_row)
        layout.addWidget(tl_grp)

        # 목표 지점 설정
        target_grp = QGroupBox("🎯 목표 지점 클릭 명령")
        target_layout = QVBoxLayout(target_grp)
        target_layout.addWidget(QLabel(
            "트랙 맵 탭에서 랜드마크를 클릭하면\n자동으로 이동 명령이 생성됩니다."
        ))
        self._lbl_clicked_target = QLabel("선택된 목표: 없음")
        self._lbl_clicked_target.setStyleSheet("color: #ffd700;")
        target_layout.addWidget(self._lbl_clicked_target)
        layout.addWidget(target_grp)

        layout.addStretch()
        return w

    # ─── 시그널 연결 ──────────────────────────────────────────────
    def _connect_signals(self):
        self._bridge.status_received.connect(self._on_status_update)
        self._bridge.feedback_received.connect(self._on_feedback)
        self._bridge.reasoning_received.connect(self._on_reasoning)
        self._bridge.vla_status_recv.connect(self._on_vla_status)
        self._bridge.odom_received.connect(self._on_odom)
        self._bridge.obstacle_received.connect(self._on_obstacle)
        self._track_map.landmark_clicked.connect(self._on_landmark_clicked)

    # ─── 슬롯 ─────────────────────────────────────────────────────
    @Slot(str)
    def _on_status_update(self, json_str: str):
        try:
            self._latest_status = json.loads(json_str)
        except Exception:
            pass

    @Slot(str)
    def _on_feedback(self, text: str):
        self._log(f"[피드백] {text}", color="#66ccff")
        self.statusBar().showMessage(text[:80])

    @Slot(str)
    def _on_reasoning(self, text: str):
        self._log(f"[이유]\n{text}", color="#ff8888")

    @Slot(str)
    def _on_vla_status(self, text: str):
        self._log(f"[VLA] {text}", color="#aaffaa")

    @Slot(float, float, float)
    def _on_odom(self, x: float, y: float, yaw_deg: float):
        self._track_map.update_vehicle(x, y, yaw_deg)
        self._lbl_pos.setText(f"위치: ({x:.2f}, {y:.2f})")
        self._lbl_yaw.setText(f"방향: {yaw_deg:.1f}°")

    @Slot(bool, float)
    def _on_obstacle(self, detected: bool, min_dist: float):
        if detected:
            self._lbl_obs.setText(f"장애물: ⚠️ 감지됨 ({min_dist:.1f}m)")
            self._lbl_obs.setStyleSheet("color: #ff6666; font-family: monospace;")
        else:
            self._lbl_obs.setText("장애물: 없음")
            self._lbl_obs.setStyleSheet("color: #66ff66; font-family: monospace;")

    @Slot(str)
    def _on_landmark_clicked(self, target: str):
        target_names = {
            "start":       "출발점",
            "crosswalk":   "횡단보도",
            "obstacle_1":  "장애물구역-1",
            "obstacle_2":  "장애물구역-2",
        }
        name = target_names.get(target, target)
        self._lbl_clicked_target.setText(f"선택된 목표: {name}")
        cmd = f"{name}로 가줘"
        self._cmd_input.setText(cmd)
        self._log(f"[맵 클릭] 목표 선택: {name}", color="#ffd700")

    def _on_send_command(self):
        cmd = self._cmd_input.text().strip()
        if cmd:
            self._send_command(cmd)
            self._cmd_input.clear()

    def _send_command(self, cmd: str):
        self._log(f"[명령 전송] {cmd}", color="#ffcc00")
        self._node.send_mission_command(cmd)
        self.statusBar().showMessage(f"명령 전송: {cmd[:60]}")

    def _on_respawn_obstacles(self):
        self._log("[시뮬] 장애물 리스폰 요청 (ros2 run simulation_pkg load_obstable_car_node 필요)", color="#ffaa00")

    # ─── UI 갱신 ─────────────────────────────────────────────────
    def _refresh_ui(self):
        if not self._latest_status:
            return

        status = self._latest_status
        self._update_task_list(status)

        # 차선 변경 상태
        obs_info = status.get("obstacle", {})
        lc_state = obs_info.get("lc_state", "IDLE")
        in_lc = lc_state not in ("IDLE",)
        self._track_map.update_lane_change(in_lc, lc_state)
        self._lbl_lcs.setText(f"차선변경: {lc_state}")

        # 목표 업데이트
        current = status.get("current")
        if current and current.get("task_type") == "navigate_to":
            target = current.get("params", {}).get("target")
            self._track_map.update_target(target)
        else:
            self._track_map.update_target(None)

        # 장애물 min_dist 업데이트
        min_dist = obs_info.get("min_dist", 999.0)
        detected = obs_info.get("detected", False)
        if detected:
            veh = status.get("vehicle", {})
            ox = veh.get("x", 0) + 3.0  # 전방 추정
            oy = veh.get("y", 0)
            self._track_map.update_obstacles([(ox, oy, min_dist)])
        else:
            self._track_map.update_obstacles([])

    def _update_task_list(self, status: dict):
        self._task_list.clear()

        sections = [
            ("▶ 실행 중", [status["current"]] if status.get("current") else [],
             QColor(50, 150, 50)),
            ("⏳ 대기", status.get("queue", []),
             QColor(100, 100, 100)),
            ("✅ 완료", list(reversed(status.get("completed", []))),
             QColor(60, 100, 150)),
            ("❌ 실패", list(reversed(status.get("failed", []))),
             QColor(150, 50, 50)),
        ]

        for section_name, tasks, color in sections:
            if not tasks:
                continue
            # 섹션 헤더
            hdr = QListWidgetItem(section_name)
            hdr.setBackground(color.darker(180))
            hdr.setForeground(color.lighter(150))
            hdr.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            hdr.setFlags(Qt.ItemFlag.NoItemFlags)
            self._task_list.addItem(hdr)

            for t in tasks:
                if t is None:
                    continue
                t_type = t.get("task_type", "?")
                params = t.get("params", {})
                s_str  = STATUS_LABELS.get(t.get("status", "pending"), "?")
                t_id   = t.get("task_id", 0)

                desc = self._describe_task_dict(t_type, params)
                label = f"  [{t_id}] {desc}  [{s_str}]"
                if t.get("reason"):
                    label += f"\n  ↳ {t['reason'][:60]}"

                item = QListWidgetItem(label)
                item_color = STATUS_COLORS.get(t.get("status", "pending"), QColor(150, 150, 150))
                item.setForeground(item_color)
                self._task_list.addItem(item)

    @staticmethod
    def _describe_task_dict(t_type: str, params: dict) -> str:
        if t_type == "spin_in_place":
            return f"제자리 {params.get('degrees', 360)}도 회전"
        elif t_type == "navigate_to":
            target = params.get("target", "?")
            avoid  = params.get("avoid_obstacles", False)
            names  = {
                "crosswalk":     "횡단보도",
                "traffic_light": "신호등",
                "start":         "출발점",
                "obstacle_1":    "장애물구역-1",
                "obstacle_2":    "장애물구역-2",
                "mid":           "중간",
            }
            n = names.get(target, target)
            return f"{n}로 이동" + (" (회피)" if avoid else "")
        elif t_type == "stop":
            return "정지"
        elif t_type == "resume":
            return "재개"
        elif t_type == "wait":
            return f"{params.get('seconds', 1)}초 대기"
        return t_type

    def _log(self, text: str, color: str = "#c9d1d9"):
        ts = time.strftime("%H:%M:%S")
        self._log_text.setTextColor(QColor(color))
        self._log_text.append(f"[{ts}] {text}")
        # 스크롤을 최하단으로
        cursor = self._log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._log_text.setTextCursor(cursor)

    # ─── 다크 테마 ────────────────────────────────────────────────
    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #13171f;
                color: #c9d1d9;
            }
            QGroupBox {
                border: 1px solid #2a3a4a;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 4px;
                font-weight: bold;
                color: #8ab4d4;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QLineEdit {
                background: #1a1e2a;
                border: 1px solid #2a3a5a;
                border-radius: 4px;
                padding: 4px 8px;
                color: #c9d1d9;
            }
            QLineEdit:focus { border-color: #3a6496; }
            QPushButton {
                background: #1e2a3a;
                border: 1px solid #2a4a6a;
                border-radius: 4px;
                padding: 4px 10px;
                color: #aacce8;
            }
            QPushButton:hover  { background: #253a4a; }
            QPushButton:pressed{ background: #1a2a3a; }
            QListWidget, QTextEdit, QTableWidget {
                background: #0d1117;
                border: 1px solid #2a3a4a;
            }
            QTabWidget::pane  { border: 1px solid #2a3a4a; }
            QTabBar::tab {
                background: #1a2030;
                color: #8ab4d4;
                padding: 6px 14px;
                border: 1px solid #2a3a4a;
            }
            QTabBar::tab:selected {
                background: #1e2a40;
                color: #ffffff;
            }
            QStatusBar { color: #8ab4d4; }
            QLabel { color: #aabbcc; }
            QComboBox {
                background: #1a1e2a;
                border: 1px solid #2a3a5a;
                border-radius: 4px;
                padding: 2px 6px;
                color: #c9d1d9;
            }
        """)


# ─── 진입점 ──────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)

    bridge   = RosBridge()
    ros_node = GUINode(bridge)

    # ROS spin을 별도 스레드에서 실행
    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()

    app = QApplication(sys.argv)
    app.setApplicationName("자율주행 미션 제어 GUI")

    window = MissionGUIWindow(ros_node, bridge)
    window.show()

    exit_code = app.exec()

    ros_node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

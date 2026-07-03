#!/usr/bin/env python3
"""
vla_gui.py — VLA 주행 실시간 명령 GUI (PySide6)

매번 `ros2 topic pub` 하던 걸 대체. 두 가지 입력:
  ① 자연어 입력창 → /nl_command (브레인 Qwen3-VL이 해석해 표준명령 발행) — 브레인 노드 필요
  ② 빠른 명령 버튼 → /vla/command (drive 노드로 직접) — 브레인 없어도 동작
실시간 로그에 실제 실행되는 /vla/command 를 에코.

실행: python3 vla_gui.py   (sim + vla_lora_drive_node 가동 상태에서. 자연어는 vla_brain_node 도)
"""
import sys
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSHistoryPolicy,
                       QoSDurabilityPolicy, QoSReliabilityPolicy)
from std_msgs.msg import String

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QGroupBox, QStatusBar,
)
from PySide6.QtCore import Qt, Signal, QObject, Slot
from PySide6.QtGui import QFont, QTextCursor

RELIABLE = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                      history=QoSHistoryPolicy.KEEP_LAST,
                      durability=QoSDurabilityPolicy.VOLATILE, depth=1)

# (라벨, 보낼 명령) — 빠른 버튼
QUICK = [
    ("▶ 1차선 주행", "1차선 계속 돌아"),
    ("▶ 2차선 주행", "2차선 계속 돌아"),
    ("🔁 1차선 한바퀴", "1차선 1바퀴 돌아"),
    ("🔁 2차선 한바퀴", "2차선 1바퀴 돌아"),
    ("🔀 2차선으로 변경", "2차선으로 변경"),
    ("🔀 1차선으로 변경", "1차선으로 변경"),
    ("⏸ 일시정지", "일시정지"),
    ("▶ 재개", "재개"),
    ("⏹ 멈춰", "멈춰"),
    ("🛟 복구(차선스냅)", "복구"),
]


class RosBridge(QObject):
    """ROS 콜백 → Qt 시그널 (스레드 안전)."""
    cmd_echo = Signal(str)        # /vla/command 수신 에코
    env_done = Signal(str, str)   # 환경 스폰/삭제 완료 (메시지, kind)


class GuiNode(Node):
    def __init__(self, bridge: RosBridge):
        super().__init__("vla_gui")
        self._bridge = bridge
        self.pub_nl = self.create_publisher(String, "nl_command", RELIABLE)
        self.pub_cmd = self.create_publisher(String, "vla/command", RELIABLE)
        self.create_subscription(String, "vla/command",
                                 lambda m: self._bridge.cmd_echo.emit(m.data), RELIABLE)

    def send_nl(self, text):
        self.pub_nl.publish(String(data=text))

    def send_cmd(self, text):
        self.pub_cmd.publish(String(data=text))


class MainWindow(QMainWindow):
    def __init__(self, node: GuiNode, bridge: RosBridge):
        super().__init__()
        self.node = node
        self.setWindowTitle("VLA 주행 명령")
        self.resize(560, 700)
        bridge.cmd_echo.connect(self._on_echo)
        bridge.env_done.connect(self._on_env_done)
        self._bridge = bridge

        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ── 자연어 입력 ──
        nl_box = QGroupBox("자연어 명령  → /nl_command  (브레인 해석)")
        nll = QVBoxLayout(nl_box)
        row = QHBoxLayout()
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("예: 옆 차선으로 가  /  한 바퀴 돌고 멈춰  /  세 바퀴 돌아")
        self.edit.returnPressed.connect(self._send_nl)
        f = QFont(); f.setPointSize(12); self.edit.setFont(f)
        btn_send = QPushButton("전송")
        btn_send.clicked.connect(self._send_nl)
        btn_send.setMinimumWidth(72)
        row.addWidget(self.edit); row.addWidget(btn_send)
        nll.addLayout(row)
        nll.addWidget(QLabel("※ 자연어는 vla_brain_node 가 떠 있어야 동작합니다."))
        root.addWidget(nl_box)

        # ── 빠른 명령 버튼 ──
        q_box = QGroupBox("빠른 명령  → /vla/command  (drive 노드로 직접)")
        grid = QGridLayout(q_box)
        for i, (label, cmd) in enumerate(QUICK):
            b = QPushButton(label)
            b.setMinimumHeight(40)
            b.clicked.connect(lambda _=False, c=cmd: self._send_cmd(c))
            grid.addWidget(b, i // 3, i % 3)
        root.addWidget(q_box)

        # ── 환경 리스폰 (장애물·신호등 스폰/삭제 토글) ──
        env_box = QGroupBox("환경 리스폰  (gazebo 스폰/삭제)")
        env_l = QHBoxLayout(env_box)
        self.btn_obs = QPushButton("🚧 장애물차량: OFF"); self.btn_obs.setCheckable(True); self.btn_obs.setMinimumHeight(40)
        self.btn_obs.toggled.connect(lambda on: self._env_toggle("obs", on))
        self.btn_tl = QPushButton("🚦 신호등: OFF"); self.btn_tl.setCheckable(True); self.btn_tl.setMinimumHeight(40)
        self.btn_tl.toggled.connect(lambda on: self._env_toggle("tl", on))
        env_l.addWidget(self.btn_obs); env_l.addWidget(self.btn_tl)
        root.addWidget(env_box)

        # ── 로그 ──
        log_box = QGroupBox("실시간 로그")
        ll = QVBoxLayout(log_box)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setFont(QFont("monospace", 10))
        ll.addWidget(self.log)
        btn_clear = QPushButton("로그 지우기"); btn_clear.clicked.connect(self.log.clear)
        ll.addWidget(btn_clear)
        root.addWidget(log_box, 1)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("준비됨 — 자연어 입력 또는 버튼 클릭")
        self._log("GUI 시작. /nl_command(자연어)·/vla/command(버튼) 발행, /vla/command 에코 표시.", "sys")

    def _log(self, text, kind="info"):
        color = {"tx": "#1565c0", "nl": "#6a1b9a", "echo": "#2e7d32", "sys": "#777"}.get(kind, "#000")
        self.log.append(f'<span style="color:{color}">{text}</span>')
        self.log.moveCursor(QTextCursor.End)

    def _send_nl(self):
        t = self.edit.text().strip()
        if not t:
            return
        self.node.send_nl(t)
        self._log(f"⌨  자연어 → '{t}'", "nl")
        self.edit.clear()
        self.statusBar().showMessage(f"자연어 전송: {t}")

    def _send_cmd(self, cmd):
        self.node.send_cmd(cmd)
        self._log(f"🔘 직접명령 → '{cmd}'", "tx")
        self.statusBar().showMessage(f"명령 전송: {cmd}")

    @Slot(str)
    def _on_echo(self, data):
        self._log(f"     ↪ 실행(vla/command): '{data}'", "echo")

    def _env_toggle(self, kind, on):
        btn = self.btn_obs if kind == "obs" else self.btn_tl
        name = "장애물차량" if kind == "obs" else "신호등"
        icon = "🚧" if kind == "obs" else "🚦"
        btn.setText(f"{icon} {name}: {'ON' if on else 'OFF'}")
        btn.setEnabled(False)
        self._log(f"{icon} {name} {'스폰 중...' if on else '삭제 중...'}", "sys")
        self.statusBar().showMessage(f"{name} {'스폰' if on else '삭제'} 중...")

        def work():
            import subprocess
            try:
                if kind == "obs":
                    import os
                    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spawn_lane_obs.py")
                    if on:
                        # 둘 다: 원래 장애물(차도+주차) + 차도 추가 장애물
                        subprocess.run(["ros2", "run", "simulation_pkg", "load_obstable_car_node"], timeout=150, capture_output=True)
                        subprocess.run(["python3", script, "on"], timeout=60, capture_output=True)
                    else:
                        for e in ("obstacle1", "obstacle2", "obstacle3", "obstacle4", "obstacle5"):
                            subprocess.run(["ros2", "service", "call", "/delete_entity",
                                            "gazebo_msgs/srv/DeleteEntity", f"{{name: '{e}'}}"], timeout=15, capture_output=True)
                        subprocess.run(["python3", script, "off"], timeout=60, capture_output=True)
                else:
                    if on:
                        subprocess.run(["ros2", "run", "simulation_pkg", "load_traffic_light_node"],
                                       timeout=60, capture_output=True)
                    else:
                        subprocess.run(["ros2", "service", "call", "/delete_entity",
                                        "gazebo_msgs/srv/DeleteEntity", "{name: 'traffic_light_stand'}"],
                                       timeout=15, capture_output=True)
                msg = f"{icon} {name} {'스폰 완료' if on else '삭제 완료'}"
            except Exception as ex:
                msg = f"{icon} {name} 실패: {ex}"
            self._bridge.env_done.emit(msg, kind)

        threading.Thread(target=work, daemon=True).start()

    @Slot(str, str)
    def _on_env_done(self, msg, kind):
        btn = self.btn_obs if kind == "obs" else self.btn_tl
        btn.setEnabled(True)
        self._log(msg, "echo")
        self.statusBar().showMessage(msg)

    def closeEvent(self, e):
        if rclpy.ok():
            rclpy.shutdown()
        e.accept()


def main():
    rclpy.init()
    bridge = RosBridge()
    node = GuiNode(bridge)
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()

    app = QApplication(sys.argv)
    win = MainWindow(node, bridge)
    win.show()
    app.exec()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()

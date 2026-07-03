#!/usr/bin/env python3
"""
VLA Command Input Node

터미널에서 자연어 명령을 입력해 VLA Brain 으로 전송합니다.

실행:
    ros2 run qwen_vl_pkg vla_cmd_node

명령 예시:
    신호등까지 가줘          → 신호등, track 모드
    신호등 직접 가줘         → 신호등, direct 모드 (기하학 경로)
    출발점으로 돌아가         → start, direct 모드
    멈춰                     → 즉시 정지
    출발                     → 재개
    상태                     → 현재 상태 확인

단축키:
    t → 신호등까지 가줘 (track)
    d → 신호등 직접 가줘 (direct)
    s → 멈춰
    g → 출발
    q → 종료
"""

import sys
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy


SHORTCUTS = {
    "t": "신호등까지 가줘",
    "d": "신호등 직접 가줘",
    "s": "멈춰",
    "g": "출발",
    "1": "신호등까지 가줘",
    "2": "신호등 직접 가줘",
    "3": "출발점으로 돌아가",
    "4": "장애물1 직접 가줘",
    "5": "장애물2 직접 가줘",
}

HELP_TEXT = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  VLA 명령 입력 노드
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  단축키:
    [1] 신호등까지 가줘 (track)
    [2] 신호등 직접 가줘 (direct)
    [3] 출발점으로 돌아가
    [4] 장애물1 직접 가줘
    [5] 장애물2 직접 가줘
    [s] 멈춰
    [g] 출발/재개
    [q] 종료

  자유 입력도 가능:
    → "신호등까지 가줘"
    → "신호등 직접 가줘"
    → "멈춰"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


class VLACmdNode(Node):
    def __init__(self):
        super().__init__("vla_cmd_node")

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.pub = self.create_publisher(String, "vla/goal_cmd", qos)

        # 상태 수신
        self.create_subscription(String, "vla/status", self._status_cb, qos)

        print(HELP_TEXT)
        threading.Thread(target=self._input_loop, daemon=True).start()

    def _status_cb(self, msg: String):
        print(f"\n  [VLA Status] {msg.data}")

    def _input_loop(self):
        while rclpy.ok():
            try:
                raw = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not raw:
                continue
            if raw.lower() == "q":
                print("종료합니다.")
                break

            # 도움말
            if raw.lower() in ("help", "h", "도움말"):
                print(HELP_TEXT)
                continue

            # 단축키 처리
            cmd = SHORTCUTS.get(raw.lower(), raw)

            msg = String()
            msg.data = cmd
            self.pub.publish(msg)
            print(f"  ✉ 전송: '{cmd}'")


def main(args=None):
    rclpy.init(args=args)
    node = VLACmdNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

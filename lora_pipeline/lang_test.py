#!/usr/bin/env python3
"""C 검증 — 자유 자연어를 /nl_command로 보내고 브레인이 변환한 표준 vla/command 확인."""
import time, rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

CASES = [
    "한 바퀴만 돌고 멈춰",
    "2차선으로 바꿔줘",
    "옆 차선으로 가",
    "계속 달려",
    "세 바퀴 돌아",
    "그만 서",
    "다시 1차선으로 돌아와",
]
rclpy.init(); n = Node("langtest")
q = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE)
got = {"cmd": None}
n.create_subscription(String, "vla/command", lambda m: got.__setitem__("cmd", m.data), q)
pub = n.create_publisher(String, "nl_command", q)
tw = time.time()
while pub.get_subscription_count() < 1 and time.time()-tw < 10:
    rclpy.spin_once(n, timeout_sec=0.1)
print(f"브레인 구독 연결: {pub.get_subscription_count()>=1}")
for c in CASES:
    got["cmd"] = None
    pub.publish(String(data=c))
    t0 = time.time()
    while got["cmd"] is None and time.time()-t0 < 8:
        rclpy.spin_once(n, timeout_sec=0.1)
    print(f"  '{c}'  →  {got['cmd']}")
n.destroy_node(); rclpy.shutdown()

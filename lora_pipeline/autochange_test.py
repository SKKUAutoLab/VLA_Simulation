#!/usr/bin/env python3
"""자동 차선전환 검증 — 차를 2차선에 놓고 '1차선 주행'을 주면 자동으로 1차선으로 건너오는지."""
import math, json, os, time, rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from gazebo_msgs.srv import SetEntityState
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

L1 = [(float(a), float(b)) for a, b in json.load(open(os.path.expanduser("~/track_gt_lane1_demo.json")))["centerline_world"]]
L0 = [(float(a), float(b)) for a, b in json.load(open(os.path.expanduser("~/track_gt_lane0_demo.json")))["centerline_world"]]
cur = {"x": 0, "y": 0}
d = lambda cl: min(math.dist((cur["x"], cur["y"]), c) for c in cl)
rclpy.init(); n = Node("autochg")
c = n.create_client(SetEntityState, "/gazebo/set_entity_state"); c.wait_for_service(timeout_sec=10)
n.create_subscription(Odometry, "/odom", lambda m: cur.update(x=m.pose.pose.position.x, y=m.pose.pose.position.y),
                      QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT))
pub = n.create_publisher(String, "vla/command", QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE))
tw = time.time()
while pub.get_subscription_count() < 1 and time.time()-tw < 5:
    rclpy.spin_once(n, timeout_sec=0.1)
# 2차선 idx150에 배치
i0 = 150; x0, y0 = L0[i0]; nx, ny = L0[(i0+1) % len(L0)]; yaw = math.atan2(ny-y0, nx-x0)+math.pi/2
r = SetEntityState.Request(); r.state.name = "ego_vehicle"
r.state.pose.position.x = x0; r.state.pose.position.y = y0; r.state.pose.position.z = 0.05
r.state.pose.orientation.z = math.sin(yaw/2); r.state.pose.orientation.w = math.cos(yaw/2); r.state.reference_frame = "world"
rclpy.spin_until_future_complete(n, c.call_async(r), timeout_sec=2)
time.sleep(0.5)
for _ in range(5): rclpy.spin_once(n, timeout_sec=0.1)
print(f"시작(2차선 배치): 1차선까지 {d(L1):.2f}m / 2차선까지 {d(L0):.2f}m")
print("명령: '1차선 계속 돌아' (차는 2차선에 있음 → 자동변경 기대)")
for _ in range(3): pub.publish(String(data="1차선 계속 돌아")); time.sleep(0.15)
t0 = time.time()
while time.time()-t0 < 16:
    rclpy.spin_once(n, timeout_sec=0.1)
    if int((time.time()-t0)) % 4 == 0:
        print(f"  [{time.time()-t0:4.1f}s] 1차선까지 {d(L1):.2f}m / 2차선까지 {d(L0):.2f}m"); time.sleep(1)
res = "✅ 1차선으로 자동 전환 성공" if d(L1) < d(L0) and d(L1) < 1.0 else "❌ 여전히 2차선/미전환"
print(res)
pub.publish(String(data="멈춰")); n.destroy_node(); rclpy.shutdown()

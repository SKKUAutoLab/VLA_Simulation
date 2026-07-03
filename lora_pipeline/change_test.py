#!/usr/bin/env python3
"""차선변경 테스트 — 1차선 출발→주행→'2차선으로 변경'→'1차선으로 변경' 순서로
명령 보내며 횡오프셋(+안쪽, 1차선≈0 / 2차선≈-2.8) 추이 출력."""
import math, json, os, time, rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from gazebo_msgs.srv import SetEntityState
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

inner = [(float(a), float(b)) for a, b in json.load(open(os.path.expanduser("~/track_gt_manual.json")))["centerline_world"]]
N = len(inner); cx = sum(p[0] for p in inner)/N; cy = sum(p[1] for p in inner)/N
cur = {"off": None}


def soff(x, y):
    i = min(range(N), key=lambda k: (inner[k][0]-x)**2+(inner[k][1]-y)**2)
    ix, iy = inner[i]; d = math.dist((x, y), (ix, iy)); inw = (cx-ix)*(x-ix)+(cy-iy)*(y-iy)
    return d if inw > 0 else -d


rclpy.init(); n = Node("chgtest")
c = n.create_client(SetEntityState, "/gazebo/set_entity_state"); c.wait_for_service(timeout_sec=10)
i0 = 150; x0, y0 = inner[i0]; nx, ny = inner[(i0+1) % N]; yaw = math.atan2(ny-y0, nx-x0)+math.pi/2
r = SetEntityState.Request(); r.state.name = "ego_vehicle"
r.state.pose.position.x = x0; r.state.pose.position.y = y0; r.state.pose.position.z = 0.05
r.state.pose.orientation.z = math.sin(yaw/2); r.state.pose.orientation.w = math.cos(yaw/2); r.state.reference_frame = "world"
rclpy.spin_until_future_complete(n, c.call_async(r), timeout_sec=2)
pub = n.create_publisher(String, "vla/command", QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE))
n.create_subscription(Odometry, "/odom", lambda m: cur.__setitem__("off", soff(m.pose.pose.position.x, m.pose.pose.position.y)),
                      QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT))
tw = time.time()
while pub.get_subscription_count() < 1 and time.time()-tw < 5: rclpy.spin_once(n, timeout_sec=0.1)


def send(cmd):
    for _ in range(3): pub.publish(String(data=cmd)); time.sleep(0.15)


def watch(sec, tag):
    t0 = time.time(); offs = []
    while time.time()-t0 < sec:
        rclpy.spin_once(n, timeout_sec=0.1)
        if cur["off"] is not None: offs.append(cur["off"])
    if offs:
        print(f"  [{tag}] 횡오프셋 끝 {offs[-1]:+.2f}m (평균 {sum(offs)/len(offs):+.2f}) | 1차선≈0, 2차선≈-2.8")


print("1) 1차선 주행 시작"); send("1차선 계속 돌아"); watch(6, "1차선 주행")
print("2) '2차선으로 변경' 전송"); send("2차선으로 변경"); watch(9, "변경 후")
print("3) '1차선으로 변경' 전송"); send("1차선으로 변경"); watch(9, "복귀 후")
send("멈춰"); n.destroy_node(); rclpy.shutdown()

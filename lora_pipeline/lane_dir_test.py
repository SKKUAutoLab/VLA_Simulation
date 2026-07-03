#!/usr/bin/env python3
"""차선·방향 진단 — 각 차선 출발점에 정확히 놓고 keep 명령 후, 실제로 어느 차선·어느 방향으로 도는지 측정."""
import math, json, os, time, rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from gazebo_msgs.srv import SetEntityState
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

L1 = [(float(a), float(b)) for a, b in json.load(open(os.path.expanduser("~/track_gt_lane1_demo.json")))["centerline_world"]]  # 1차선(inner)
L0 = [(float(a), float(b)) for a, b in json.load(open(os.path.expanduser("~/track_gt_lane0_demo.json")))["centerline_world"]]  # 2차선(outer)


def nearest(cl, x, y):
    return min(range(len(cl)), key=lambda k: (cl[k][0]-x)**2+(cl[k][1]-y)**2)


def dist_to(cl, x, y):
    i = nearest(cl, x, y); return math.dist((x, y), cl[i])


cur = {"x": 0, "y": 0}
rclpy.init(); n = Node("lanedir")
c = n.create_client(SetEntityState, "/gazebo/set_entity_state"); c.wait_for_service(timeout_sec=10)
n.create_subscription(Odometry, "/odom", lambda m: cur.update(x=m.pose.pose.position.x, y=m.pose.pose.position.y),
                      QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT))
pub = n.create_publisher(String, "vla/command", QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE))
tw = time.time()
while pub.get_subscription_count() < 1 and time.time()-tw < 5:
    rclpy.spin_once(n, timeout_sec=0.1)


def teleport(cl, i0):
    x0, y0 = cl[i0]; nx, ny = cl[(i0+1) % len(cl)]; yaw = math.atan2(ny-y0, nx-x0)+math.pi/2
    r = SetEntityState.Request(); r.state.name = "ego_vehicle"
    r.state.pose.position.x = x0; r.state.pose.position.y = y0; r.state.pose.position.z = 0.05
    r.state.pose.orientation.z = math.sin(yaw/2); r.state.pose.orientation.w = math.cos(yaw/2); r.state.reference_frame = "world"
    rclpy.spin_until_future_complete(n, c.call_async(r), timeout_sec=2)


def test(name, cl, lane_id_for_dir):
    teleport(cl, 150)
    pub.publish(String(data="멈춰")); time.sleep(0.5)
    for _ in range(3):
        pub.publish(String(data=f"{name} 계속 돌아")); time.sleep(0.15)
    idxs = []; t0 = time.time()
    while time.time()-t0 < 14:
        rclpy.spin_once(n, timeout_sec=0.1)
        idxs.append(nearest(cl, cur["x"], cur["y"]))
    d1 = dist_to(L1, cur["x"], cur["y"]); d0 = dist_to(L0, cur["x"], cur["y"])
    # 방향: 출발 150 기준 인덱스 증감(±N 래핑 고려)
    N = len(cl); di = idxs[-1]-150
    if di > N/2: di -= N
    if di < -N/2: di += N
    drive_lane = "1차선(inner)" if d1 < d0 else "2차선(outer)"
    direction = "정방향(전진)" if di > 0 else ("역방향(역주행!)" if di < 0 else "정지/제자리")
    print(f"\n=== 명령 '{name} 계속 돌아' (출발: {name} 차선) ===")
    print(f"  14초 후 위치: 1차선까지 {d1:.2f}m / 2차선까지 {d0:.2f}m → 실제 주행차선: {drive_lane}")
    print(f"  인덱스 변화 {di:+d} → 방향: {direction}")


test("1차선", L1, 0)
test("2차선", L0, 1)
pub.publish(String(data="멈춰"))
n.destroy_node(); rclpy.shutdown()

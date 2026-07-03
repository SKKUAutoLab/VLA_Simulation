#!/usr/bin/env python3
"""GUI 자연어 경로 검증 — /nl_command(=GUI 입력창과 동일)로 대화형 문장을 보내
브레인 해석(vla/command) + 차량 실제 반응(횡오프셋/속도)을 함께 관찰."""
import math, json, os, time, rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from gazebo_msgs.srv import SetEntityState
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

lane = [(float(a), float(b)) for a, b in json.load(open(os.path.expanduser("~/track_gt_lane1_demo.json")))["centerline_world"]]
N = len(lane); cx = sum(p[0] for p in lane)/N; cy = sum(p[1] for p in lane)/N
cur = {"off": None, "v": 0.0}


def soff(x, y):
    i = min(range(N), key=lambda k: (lane[k][0]-x)**2+(lane[k][1]-y)**2)
    ix, iy = lane[i]; d = math.dist((x, y), (ix, iy)); inw = (cx-ix)*(x-ix)+(cy-iy)*(y-iy)
    return d if inw > 0 else -d


rclpy.init(); n = Node("nldrv")
c = n.create_client(SetEntityState, "/gazebo/set_entity_state"); c.wait_for_service(timeout_sec=10)
i0 = 150; x0, y0 = lane[i0]; nx, ny = lane[(i0+1) % N]; yaw = math.atan2(ny-y0, nx-x0)+math.pi/2
r = SetEntityState.Request(); r.state.name = "ego_vehicle"
r.state.pose.position.x = x0; r.state.pose.position.y = y0; r.state.pose.position.z = 0.05
r.state.pose.orientation.z = math.sin(yaw/2); r.state.pose.orientation.w = math.cos(yaw/2); r.state.reference_frame = "world"
rclpy.spin_until_future_complete(n, c.call_async(r), timeout_sec=2)


def od(m):
    p = m.pose.pose.position; v = m.twist.twist.linear
    cur["off"] = soff(p.x, p.y); cur["v"] = math.hypot(v.x, v.y)


n.create_subscription(Odometry, "/odom", od, QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT))
q = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE)
last = {"cmd": None}
n.create_subscription(String, "vla/command", lambda m: last.__setitem__("cmd", m.data), q)
pub = n.create_publisher(String, "nl_command", q)
tw = time.time()
while pub.get_subscription_count() < 1 and time.time()-tw < 5:
    rclpy.spin_once(n, timeout_sec=0.1)


def run(phrase, watch_s):
    last["cmd"] = None
    pub.publish(String(data=phrase))
    t0 = time.time(); seen = None
    while time.time()-t0 < watch_s:
        rclpy.spin_once(n, timeout_sec=0.1)
        if last["cmd"] and seen is None:
            seen = last["cmd"]
    print(f"\n💬 자연어: '{phrase}'")
    print(f"   → 브레인 해석(vla/command): {seen}")
    print(f"   → 차량: 횡오프셋 {cur['off']:+.2f}m (1차선0/2차선-2.8), 속도 {cur['v']:.2f}m/s")


print("="*55)
run("1차선으로 쭉 달려줘", 7)
run("옆 차선으로 바꿔줘", 15)
run("좋아 이제 멈춰", 4)
print("\n" + "="*55)
n.destroy_node(); rclpy.shutdown()

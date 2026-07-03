#!/usr/bin/env python3
"""한바퀴 돌며 차선중심선 인덱스별 이탈 기록 → 최악 구간 top 보고(코너 진단)."""
import argparse, math, json, time, os, statistics as st
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from gazebo_msgs.srv import SetEntityState
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

LANE_FILE = {0: "~/track_gt_lane0_demo.json", 1: "~/track_gt_lane1_demo.json"}
CMD = {0: "1차선 한바퀴 돌아", 1: "2차선 한바퀴 돌아"}

ap = argparse.ArgumentParser(); ap.add_argument("--lane", type=int, default=0); ap.add_argument("--secs", type=float, default=140)
a, _ = ap.parse_known_args()
cl = [(float(x), float(y)) for x, y in json.load(open(os.path.expanduser(LANE_FILE[a.lane])))["centerline_world"]]
N = len(cl)
rclpy.init(); n = Node("findworst")
c = n.create_client(SetEntityState, "/gazebo/set_entity_state"); c.wait_for_service(timeout_sec=10)
i0 = 150; x0, y0 = cl[i0]; nx, ny = cl[(i0+1) % N]; yaw = math.atan2(ny-y0, nx-x0)+math.pi/2
r = SetEntityState.Request(); r.state.name = "ego_vehicle"
r.state.pose.position.x = x0; r.state.pose.position.y = y0; r.state.pose.position.z = 0.05
r.state.pose.orientation.z = math.sin(yaw/2); r.state.pose.orientation.w = math.cos(yaw/2); r.state.reference_frame = "world"
rclpy.spin_until_future_complete(n, c.call_async(r), timeout_sec=2)
pub = n.create_publisher(String, "vla/command", QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE))
time.sleep(0.6); pub.publish(String(data=CMD[a.lane])); time.sleep(0.4)
byidx = {}
def cb(m):
    p = (m.pose.pose.position.x, m.pose.pose.position.y)
    i = min(range(N), key=lambda k: (cl[k][0]-p[0])**2+(cl[k][1]-p[1])**2)
    byidx.setdefault(i//10, []).append(math.dist(p, cl[i]))
n.create_subscription(Odometry, "/odom", cb, QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT))
t0 = time.time()
while time.time()-t0 < a.secs:
    rclpy.spin_once(n, timeout_sec=0.1)
pub.publish(String(data="멈춰"))
rows = [(seg, max(v), st.mean(v)) for seg, v in byidx.items()]
rows.sort(key=lambda x: -x[1])
print(f"=== LANE{a.lane} 최악 구간 top8 (seg=인덱스/10, 즉 seg*10 부근) ===")
for seg, mx, me in rows[:8]:
    print(f"  seg{seg:>3} (idx~{seg*10}): 최대 {mx:.2f}m 평균 {me:.2f}m")
n.destroy_node(); rclpy.shutdown()

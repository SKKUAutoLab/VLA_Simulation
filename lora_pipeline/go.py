#!/usr/bin/env python3
"""차를 지정 차선 출발점에 텔레포트하고 'N바퀴 돌아' 명령 전송.
사용: python3 go.py <lane:1|2> [laps]   예) python3 go.py 2     python3 go.py 1 2
주행 노드(vla_lora_drive_node / vla_wp_drive_node / wp_drive_node 중 하나)가 떠 있어야 함."""
import sys, math, json, os, time, rclpy
from rclpy.node import Node
from std_msgs.msg import String
from gazebo_msgs.srv import SetEntityState
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

lane = sys.argv[1] if len(sys.argv) > 1 else "2"
laps = sys.argv[2] if len(sys.argv) > 2 else "1"
# 1차선=중심선(lane1_demo), 2차선=바깥(lane0_demo)
FILE = os.path.expanduser("~/track_gt_lane1_demo.json" if lane == "1" else "~/track_gt_lane0_demo.json")
cl = [(float(x), float(y)) for x, y in json.load(open(FILE))["centerline_world"]]
N = len(cl)
rclpy.init(); n = Node("go")
c = n.create_client(SetEntityState, "/gazebo/set_entity_state"); c.wait_for_service(timeout_sec=10)
i0 = 150; x0, y0 = cl[i0]; nx, ny = cl[(i0+1) % N]; yaw = math.atan2(ny-y0, nx-x0)+math.pi/2
r = SetEntityState.Request(); r.state.name = "ego_vehicle"
r.state.pose.position.x = x0; r.state.pose.position.y = y0; r.state.pose.position.z = 0.05
r.state.pose.orientation.z = math.sin(yaw/2); r.state.pose.orientation.w = math.cos(yaw/2); r.state.reference_frame = "world"
rclpy.spin_until_future_complete(n, c.call_async(r), timeout_sec=2)
pub = n.create_publisher(String, "vla/command", QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE))
tw = time.time()
while pub.get_subscription_count() < 1 and time.time()-tw < 5:
    rclpy.spin_once(n, timeout_sec=0.1)
time.sleep(0.3)
cmd = f"{lane}차선 {laps}바퀴 돌아"
for _ in range(3):
    pub.publish(String(data=cmd)); time.sleep(0.2)
print(f"✓ {lane}차선 출발점 텔레포트 + '{cmd}' 전송")
n.destroy_node(); rclpy.shutdown()

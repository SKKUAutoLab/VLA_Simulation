#!/usr/bin/env python3
"""한바퀴 검증 — '1차선 1바퀴 돌아' 보내고, 한 바퀴 후 자동정지하는지 측정."""
import math, json, os, time, rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from gazebo_msgs.srv import SetEntityState
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

lane = [(float(a), float(b)) for a, b in json.load(open(os.path.expanduser("~/track_gt_lane1_demo.json")))["centerline_world"]]
N = len(lane)
rclpy.init(); n = Node("laptest")
c = n.create_client(SetEntityState, "/gazebo/set_entity_state"); c.wait_for_service(timeout_sec=10)
i0 = 150; x0, y0 = lane[i0]; nx, ny = lane[(i0+1) % N]; yaw = math.atan2(ny-y0, nx-x0)+math.pi/2
r = SetEntityState.Request(); r.state.name = "ego_vehicle"
r.state.pose.position.x = x0; r.state.pose.position.y = y0; r.state.pose.position.z = 0.05
r.state.pose.orientation.z = math.sin(yaw/2); r.state.pose.orientation.w = math.cos(yaw/2); r.state.reference_frame = "world"
rclpy.spin_until_future_complete(n, c.call_async(r), timeout_sec=2)

st = {"x": x0, "y": y0, "v": 0.0, "path": 0.0, "px": x0, "py": y0, "maxd": 0.0}


def od(m):
    p = m.pose.pose.position; v = m.twist.twist.linear
    st["v"] = math.hypot(v.x, v.y)
    st["path"] += math.dist((p.x, p.y), (st["px"], st["py"])); st["px"], st["py"] = p.x, p.y
    st["x"], st["y"] = p.x, p.y
    st["maxd"] = max(st["maxd"], math.dist((p.x, p.y), (x0, y0)))


n.create_subscription(Odometry, "/odom", od, QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT))
pub = n.create_publisher(String, "vla/command", QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE))
tw = time.time()
while pub.get_subscription_count() < 1 and time.time()-tw < 5:
    rclpy.spin_once(n, timeout_sec=0.1)
for _ in range(3):
    pub.publish(String(data="1차선 1바퀴 돌아")); time.sleep(0.15)
print("'1차선 1바퀴 돌아' 전송. 한바퀴 후 자동정지 관찰(최대 160초)...")
t0 = time.time(); moved = False; stop_t = None; result = None
while time.time()-t0 < 160:
    rclpy.spin_once(n, timeout_sec=0.1)
    el = time.time()-t0
    if st["v"] > 0.3:
        moved = True
    # 충분히 멀어졌다(>20m) 출발점 복귀(<3m) 후 정지(<0.1)면 한바퀴 완주정지
    near_start = math.dist((st["x"], st["y"]), (x0, y0)) < 3.0
    if moved and st["maxd"] > 20 and near_start and st["v"] < 0.1:
        if stop_t is None:
            stop_t = time.time()
        elif time.time()-stop_t > 2.0:
            result = f"✅ 한바퀴 완주 후 자동정지 (소요 {el:.0f}s, 주행거리 {st['path']:.0f}m, 최대이격 {st['maxd']:.0f}m)"
            break
    else:
        stop_t = None
    if int(el) % 15 == 0 and el > 1:
        print(f"  [{el:3.0f}s] 주행거리 {st['path']:5.0f}m | 출발점거리 {math.dist((st['x'],st['y']),(x0,y0)):4.1f}m | 속도 {st['v']:.2f}")
        time.sleep(1)
print(result or f"⏱ 160초 내 미완(주행 {st['path']:.0f}m, 최대이격 {st['maxd']:.0f}m, 속도 {st['v']:.2f})")
pub.publish(String(data="멈춰")); n.destroy_node(); rclpy.shutdown()

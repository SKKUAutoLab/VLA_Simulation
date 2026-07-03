#!/usr/bin/env python3
"""통합 폐루프 — 1차선 주행 중 전방에 장애물 출현 → 브레인 제로샷 감지 → 차선변경 회피 → 통과.
drive 노드(주행) + brain 노드(장면추론) 동시 가동 상태에서 실행."""
import math, json, os, time, rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from gazebo_msgs.srv import SetEntityState, SpawnEntity, DeleteEntity
from geometry_msgs.msg import Pose
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

lane = [(float(a), float(b)) for a, b in json.load(open(os.path.expanduser("~/track_gt_lane1_demo.json")))["centerline_world"]]
N = len(lane); cx = sum(p[0] for p in lane)/N; cy = sum(p[1] for p in lane)/N
cur = {"off": None, "x": 0, "y": 0}


def soff(x, y):
    i = min(range(N), key=lambda k: (lane[k][0]-x)**2+(lane[k][1]-y)**2)
    ix, iy = lane[i]; d = math.dist((x, y), (ix, iy)); inw = (cx-ix)*(x-ix)+(cy-iy)*(y-iy)
    return d if inw > 0 else -d


rclpy.init(); n = Node("intdemo")
setc = n.create_client(SetEntityState, "/gazebo/set_entity_state"); setc.wait_for_service(timeout_sec=10)
spc = n.create_client(SpawnEntity, "/spawn_entity"); spc.wait_for_service(timeout_sec=10)
delc = n.create_client(DeleteEntity, "/delete_entity"); delc.wait_for_service(timeout_sec=10)
q = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE)
be = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
pub = n.create_publisher(String, "vla/command", q)


def odcb(m):
    p = m.pose.pose.position; cur["off"] = soff(p.x, p.y); cur["x"] = p.x; cur["y"] = p.y


n.create_subscription(Odometry, "/odom", odcb, be)
emits = []
n.create_subscription(String, "vla/command", lambda m: emits.append((round(time.time()-t0, 1), m.data)), q)

# 1) ego 1차선 idx150 텔레포트
i0 = 150; x0, y0 = lane[i0]; nx, ny = lane[(i0+1) % N]; yaw = math.atan2(ny-y0, nx-x0)
r = SetEntityState.Request(); r.state.name = "ego_vehicle"
r.state.pose.position.x = x0; r.state.pose.position.y = y0; r.state.pose.position.z = 0.05
ey = yaw+math.pi/2; r.state.pose.orientation.z = math.sin(ey/2); r.state.pose.orientation.w = math.cos(ey/2); r.state.reference_frame = "world"
rclpy.spin_until_future_complete(n, setc.call_async(r), timeout_sec=2)

# 2) 전방 ~16m 1차선 위에 장애물 차량 스폰
d = 0.0; j = i0
while d < 26.0:
    a = lane[j % N]; b = lane[(j+1) % N]; d += math.dist(a, b); j += 1
ox, oy = lane[j % N]; obyaw = math.atan2(lane[(j+1) % N][1]-oy, lane[(j+1) % N][0]-ox)
sp = SpawnEntity.Request(); sp.name = "hazard_obj"
sp.xml = open(os.path.expanduser("~/.gazebo/models/hatchback_red/model.sdf")).read()
p = Pose(); p.position.x = ox; p.position.y = oy; p.position.z = 0.05
p.orientation.z = math.sin(obyaw/2); p.orientation.w = math.cos(obyaw/2); sp.initial_pose = p
rclpy.spin_until_future_complete(n, spc.call_async(sp), timeout_sec=5)

t0 = time.time()
tw = time.time()
while pub.get_subscription_count() < 1 and time.time()-tw < 5:
    rclpy.spin_once(n, timeout_sec=0.1)
print(f"장애물 1차선 전방 26m ({ox:.1f},{oy:.1f}). 주행 시작 → 브레인이 감지·회피하는지 관찰")
# 3) 1차선 주행 시작
for _ in range(3): pub.publish(String(data="1차선 계속 돌아")); time.sleep(0.15)

# 4) 35초 관찰: 오프셋 + 장애물까지 거리 추이, 브레인 명령
log = []
t1 = time.time()
while time.time()-t1 < 48:
    rclpy.spin_once(n, timeout_sec=0.1)
    if cur["off"] is not None:
        dist = math.dist((cur["x"], cur["y"]), (ox, oy))
        log.append((round(time.time()-t0, 1), cur["off"], dist))
print("\n브레인 발행 명령:")
for ts, c in emits:
    print(f"  [{ts}s] '{c}'")
print("\n오프셋·장애물거리 추이(2초 간격):")
last = -9
for ts, off, dist in log:
    if ts-last >= 2:
        last = ts; print(f"  [{ts:4.1f}s] 횡오프셋 {off:+.2f}m (1차선0/2차선-2.8) | 장애물까지 {dist:4.1f}m")
mind = min(d for _, _, d in log) if log else 99
print(f"\n장애물 최근접 {mind:.1f}m | 회피판정: {'성공(2차선으로 우회)' if any(o < -1.8 for _, o, _ in log) else '미회피'}")
pub.publish(String(data="멈춰"))
dr = DeleteEntity.Request(); dr.name = "hazard_obj"
rclpy.spin_until_future_complete(n, delc.call_async(dr), timeout_sec=3)
n.destroy_node(); rclpy.shutdown()

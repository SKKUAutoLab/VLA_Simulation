#!/usr/bin/env python3
"""D 장애물 검증 — 1차선 전방에 차량 스폰, 브레인이 카메라로 제로샷 '장애물' 판단→회피명령 발행 관찰.
사용: python3 d_obstacle_test.py [redlight]  (인자 redlight면 traffic 신호등 스폰)"""
import math, json, os, sys, time, rclpy
from rclpy.node import Node
from std_msgs.msg import String
from gazebo_msgs.srv import SetEntityState, SpawnEntity, DeleteEntity
from geometry_msgs.msg import Pose
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

MODE = sys.argv[1] if len(sys.argv) > 1 else "obstacle"
MODEL = "redlight" if MODE == "redlight" else "hatchback_red"
SDF = os.path.expanduser(f"~/.gazebo/models/{MODEL}/model.sdf")
if not os.path.exists(SDF):
    SDF = f"/home/autolab/VLA_simulation/src/simulation_pkg/models/{MODEL}/model.sdf"
xml = open(SDF).read()
lane = [(float(a), float(b)) for a, b in json.load(open(os.path.expanduser("~/track_gt_lane1_demo.json")))["centerline_world"]]
N = len(lane)

rclpy.init(); n = Node("dtest")
setc = n.create_client(SetEntityState, "/gazebo/set_entity_state"); setc.wait_for_service(timeout_sec=10)
spc = n.create_client(SpawnEntity, "/spawn_entity"); spc.wait_for_service(timeout_sec=10)
delc = n.create_client(DeleteEntity, "/delete_entity"); delc.wait_for_service(timeout_sec=10)

i0 = 150; x0, y0 = lane[i0]; nx, ny = lane[(i0+1) % N]; yaw = math.atan2(ny-y0, nx-x0)
# ego 텔레포트 (yaw+pi/2 = prius 전방규약)
r = SetEntityState.Request(); r.state.name = "ego_vehicle"
r.state.pose.position.x = x0; r.state.pose.position.y = y0; r.state.pose.position.z = 0.05
eyaw = yaw + math.pi/2
r.state.pose.orientation.z = math.sin(eyaw/2); r.state.pose.orientation.w = math.cos(eyaw/2); r.state.reference_frame = "world"
rclpy.spin_until_future_complete(n, setc.call_async(r), timeout_sec=2)

# 전방 ~9m 지점 인덱스 찾기
d = 0.0; j = i0
while d < 9.0:
    a = lane[j % N]; b = lane[(j+1) % N]; d += math.dist(a, b); j += 1
ox, oy = lane[j % N]
print(f"ego idx{i0} ({x0:.1f},{y0:.1f}) / 장애물 idx{j%N} ({ox:.1f},{oy:.1f}) 거리~{d:.1f}m, 모델={MODEL}")

# 장애물 스폰 (트랙 방향 정렬)
sp = SpawnEntity.Request(); sp.name = "hazard_obj"; sp.xml = xml
p = Pose(); p.position.x = ox; p.position.y = oy; p.position.z = 0.05
oyaw = yaw + (0 if MODE == "obstacle" else math.pi)   # 차량은 진행방향, 신호등은 마주보게
p.orientation.z = math.sin(oyaw/2); p.orientation.w = math.cos(oyaw/2); sp.initial_pose = p
fut = spc.call_async(sp); rclpy.spin_until_future_complete(n, fut, timeout_sec=5)
print(f"스폰 결과: {fut.result().success if fut.result() else 'timeout'} {fut.result().status_message if fut.result() else ''}")

# 브레인의 vla/command 관찰
q = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE)
emits = []
n.create_subscription(String, "vla/command", lambda m: emits.append((round(time.time()-t0, 1), m.data)), q)
print("브레인 장면추론 관찰 22초 (장애물/빨간불 판단 대기)...")
t0 = time.time()
while time.time()-t0 < 22:
    rclpy.spin_once(n, timeout_sec=0.1)
for ts, c in emits:
    print(f"  [{ts}s] vla/command ← '{c}'")
if not emits:
    print("  (발행 없음 — 위험 미감지 또는 상태변화 없음)")

# 정리
dr = DeleteEntity.Request(); dr.name = "hazard_obj"
rclpy.spin_until_future_complete(n, delc.call_async(dr), timeout_sec=3)
print("장애물 제거 완료")
n.destroy_node(); rclpy.shutdown()

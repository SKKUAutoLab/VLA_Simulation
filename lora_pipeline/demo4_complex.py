#!/usr/bin/env python3
"""복잡명령 데모 — '1차선 1바퀴 돌아' + 주행 중 빨간불에서 일시정지(랩 유지)→3초 후 초록불→재개→완주.
빨간불 정지/재개는 브레인 장면추론이 자율 수행. '3초'는 신호등(빨강) 지속시간으로 구현."""
import math, json, os, time, rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from gazebo_msgs.srv import SetEntityState, SpawnEntity, DeleteEntity
from geometry_msgs.msg import Pose
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

lane = [(float(a), float(b)) for a, b in json.load(open(os.path.expanduser("~/track_gt_lane1_demo.json")))["centerline_world"]]
N = len(lane)
rclpy.init(); n = Node("demo4")
setc = n.create_client(SetEntityState, "/gazebo/set_entity_state"); setc.wait_for_service(timeout_sec=10)
spc = n.create_client(SpawnEntity, "/spawn_entity"); spc.wait_for_service(timeout_sec=10)
delc = n.create_client(DeleteEntity, "/delete_entity"); delc.wait_for_service(timeout_sec=10)

i0 = 150; x0, y0 = lane[i0]; nx, ny = lane[(i0+1) % N]; yaw = math.atan2(ny-y0, nx-x0)
r = SetEntityState.Request(); r.state.name = "ego_vehicle"
r.state.pose.position.x = x0; r.state.pose.position.y = y0; r.state.pose.position.z = 0.05
ey = yaw+math.pi/2; r.state.pose.orientation.z = math.sin(ey/2); r.state.pose.orientation.w = math.cos(ey/2); r.state.reference_frame = "world"
rclpy.spin_until_future_complete(n, setc.call_async(r), timeout_sec=2)

# 신호등 위치: 출발 전방 ~12m
d = 0.0; j = i0
while d < 12.0:
    a = lane[j % N]; b = lane[(j+1) % N]; d += math.dist(a, b); j += 1
lx, ly = lane[j % N]
st = {"x": x0, "y": y0, "v": 0.0, "maxd": 0.0}


def od(m):
    p = m.pose.pose.position; v = m.twist.twist.linear
    st["v"] = math.hypot(v.x, v.y); st["x"], st["y"] = p.x, p.y
    st["maxd"] = max(st["maxd"], math.dist((p.x, p.y), (x0, y0)))


n.create_subscription(Odometry, "/odom", od, QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT))
emits = []
q = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE)
n.create_subscription(String, "vla/command", lambda m: emits.append((round(time.time()-t0, 1), m.data)), q)
pub = n.create_publisher(String, "vla/command", q)
tw = time.time()
while pub.get_subscription_count() < 1 and time.time()-tw < 5:
    rclpy.spin_once(n, timeout_sec=0.1)

t0 = time.time()
print("명령: '1차선 1바퀴 돌아' (빨간불 정지·재개는 자율)")
for _ in range(3):
    pub.publish(String(data="1차선 1바퀴 돌아")); time.sleep(0.15)
# 빨간불 스폰
sp = SpawnEntity.Request(); sp.name = "hazard_obj"
sp.xml = open(os.path.expanduser("~/.gazebo/models/redlight/model.sdf")).read()
p = Pose(); p.position.x = lx; p.position.y = ly; p.position.z = 0.05
lyaw = yaw+math.pi; p.orientation.z = math.sin(lyaw/2); p.orientation.w = math.cos(lyaw/2); sp.initial_pose = p
rclpy.spin_until_future_complete(n, spc.call_async(sp), timeout_sec=5)
print(f"🔴 빨간불 전방 12m ({lx:.1f},{ly:.1f}) 점등")

phase = "approach"; paused_t = None; light_gone = False; done = False; ev = []
while time.time()-t0 < 170:
    rclpy.spin_once(n, timeout_sec=0.1)
    el = time.time()-t0
    dl = math.dist((st["x"], st["y"]), (lx, ly)); ds = math.dist((st["x"], st["y"]), (x0, y0))
    if phase == "approach" and st["v"] < 0.1 and dl < 9 and st["maxd"] > 3:
        phase = "paused"; paused_t = time.time(); ev.append(f"[{el:.0f}s] ⏸ 빨간불 앞 정지 (신호등까지 {dl:.1f}m)")
    elif phase == "paused" and not light_gone and time.time()-paused_t > 3.0:
        dr = DeleteEntity.Request(); dr.name = "hazard_obj"
        rclpy.spin_until_future_complete(n, delc.call_async(dr), timeout_sec=3)
        light_gone = True; ev.append(f"[{el:.0f}s] 🟢 3초 경과 → 초록불(신호등 소등)")
    elif phase == "paused" and light_gone and st["v"] > 0.3:
        phase = "resumed"; ev.append(f"[{el:.0f}s] ▶ 재개·주행 (속도 {st['v']:.2f})")
    elif phase == "resumed" and st["maxd"] > 20 and ds < 4 and st["v"] < 0.1:
        done = True; ev.append(f"[{el:.0f}s] ✅ 같은 랩 이어서 완주 후 자동정지 (출발점 {ds:.1f}m)"); break

print("\n=== 이벤트 타임라인 ===")
for e in ev:
    print(" ", e)
print("\n=== 브레인 발행 ===")
for ts, c in emits:
    print(f"  [{ts}s] '{c}'")
if not done:
    print(f"\n⏱ 미완(phase={phase}, 최대이격 {st['maxd']:.0f}m, 속도 {st['v']:.2f})")
pub.publish(String(data="멈춰"))
if not light_gone:
    dr = DeleteEntity.Request(); dr.name = "hazard_obj"; rclpy.spin_until_future_complete(n, delc.call_async(dr), timeout_sec=3)
n.destroy_node(); rclpy.shutdown()

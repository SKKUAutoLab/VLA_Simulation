#!/usr/bin/env python3
"""
풀랩 평가 — 지정 차선 위에 차를 놓고 '한바퀴' 무정지 주행시켜
차선중앙 최대이탈 / 완주율을 측정. (vla_cnn_drive_node 가 떠 있어야 함)
사용: python3 eval_lap.py --lane 0|1 [--secs 220] [--maxoff 0.5]
"""
import argparse, math, json, time, os
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from gazebo_msgs.srv import SetEntityState
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

# 시연-적합 라인(주석과 0.2m일치). lane0=1차선(중심선), lane1=2차선(바깥)
LANE_FILE = {0: os.path.expanduser("~/track_gt_lane1_demo.json"),
             1: os.path.expanduser("~/track_gt_lane0_demo.json")}
CMD = {0: "1차선 한바퀴 돌아", 1: "2차선 한바퀴 돌아"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", type=int, default=0)
    ap.add_argument("--secs", type=float, default=220)
    ap.add_argument("--maxoff", type=float, default=0.5)   # 라인 안밟기 기준
    ap.add_argument("--start-idx", type=int, default=150)
    a, _ = ap.parse_known_args()
    cl = [(float(x), float(y)) for x, y in json.load(open(LANE_FILE[a.lane]))["centerline_world"]]
    N = len(cl)

    rclpy.init()
    n = Node("eval_lap")
    c = n.create_client(SetEntityState, "/gazebo/set_entity_state")
    c.wait_for_service(timeout_sec=10)
    # 차선 위 출발 텔레포트
    i0 = a.start_idx
    x0, y0 = cl[i0]; nx, ny = cl[(i0+1) % N]
    yaw = math.atan2(ny-y0, nx-x0) + math.pi/2
    r = SetEntityState.Request(); r.state.name = "ego_vehicle"
    r.state.pose.position.x = x0; r.state.pose.position.y = y0; r.state.pose.position.z = 0.05
    r.state.pose.orientation.z = math.sin(yaw/2); r.state.pose.orientation.w = math.cos(yaw/2)
    r.state.reference_frame = "world"
    rclpy.spin_until_future_complete(n, c.call_async(r), timeout_sec=2)
    pub = n.create_publisher(String, "vla/command",
                             QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                                        durability=QoSDurabilityPolicy.VOLATILE))
    # 노드 구독자 디스커버리 대기 후 발행(누락 방지)
    t_wait = time.time()
    while pub.get_subscription_count() < 1 and time.time()-t_wait < 5:
        rclpy.spin_once(n, timeout_sec=0.1)
    time.sleep(0.3)
    for _ in range(3):
        pub.publish(String(data=CMD[a.lane])); time.sleep(0.2)

    # 조향 흔들림 측정용 구독
    from interfaces_pkg.msg import MotionCommand
    steers = []
    n.create_subscription(MotionCommand, "topic_control_signal",
                          lambda m: steers.append(m.steering),
                          QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                                     durability=QoSDurabilityPolicy.VOLATILE))

    offs = []; segs = set(); idxs = []
    def nidx(p):
        return min(range(N), key=lambda k: (cl[k][0]-p[0])**2 + (cl[k][1]-p[1])**2)
    def cb(m):
        p = (m.pose.pose.position.x, m.pose.pose.position.y)
        i = nidx(p); offs.append(math.dist(p, cl[i])); segs.add(i // 20); idxs.append(i)
    n.create_subscription(Odometry, "/odom", cb,
                          QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT))
    t0 = time.time(); done = False
    while time.time() - t0 < a.secs:
        rclpy.spin_once(n, timeout_sec=0.1)
        # 완주: 거의 모든 세그먼트 방문 + 시작 인덱스 근처 복귀
        if len(segs) >= 34 and idxs and abs(idxs[-1]-i0) < 12:
            done = True; break
    mx = max(offs) if offs else 99
    mean = sum(offs)/len(offs) if offs else 99
    cov = 100*len(segs)/(N//20)
    over = sum(1 for o in offs if o > a.maxoff)
    passed = done and mx < a.maxoff
    print(f"=== LANE{a.lane} 풀랩 평가 (경과 {time.time()-t0:.0f}s) ===")
    print(f"완주: {'YES' if done else 'NO'} | 세그먼트 {cov:.0f}% | 평균이탈 {mean:.2f}m 최대 {mx:.2f}m")
    print(f"라인기준 {a.maxoff}m 초과 프레임 {100*over/max(1,len(offs)):.0f}%")
    if len(steers) > 5:
        dsteer = [abs(steers[k]-steers[k-1]) for k in range(1, len(steers))]
        flips = sum(1 for k in range(1, len(steers)) if steers[k]*steers[k-1] < 0)
        print(f"조향 흔들림: 평균변화 {sum(dsteer)/len(dsteer):.2f}단계/스텝, 부호반전 {flips}회 (낮을수록 부드러움)")
    print(f"==> {'✅ PASS (라인 안밟고 완주)' if passed else '❌ FAIL'}")
    n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()

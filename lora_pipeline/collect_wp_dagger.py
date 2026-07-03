#!/usr/bin/env python3
"""
WP-DAgger 수집 — 현재 모델(wp_drive_node)이 on-policy로 주행하는 동안
차가 실제 방문하는 상태(코너 표류 포함)의 카메라+pose를 기록하고,
각 상태에 '차선 중심선 기준 정답 미래 WP'(ego좌표)를 라벨로 붙임.
→ 모델이 빠지는 OOD 코너 상태에서 라인복귀 WP를 학습(분포시프트 교정).
과대이탈(>MAXOFF)은 폐기(복구불가 garbage 방지).
저장: dataset/images/dag_*.jpg, dataset/labels_wpLdag{lane}.csv  (train_wp.py가 glob으로 흡수)
사용: 현재 wp_drive_node 떠있는 상태에서 python3 collect_wp_dagger.py --lane 0 --secs 280
"""
import argparse, math, json, time, os
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from cv_bridge import CvBridge
from gazebo_msgs.srv import SetEntityState
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

HERE = os.path.dirname(__file__)
IMG_DIR = os.path.join(HERE, "dataset", "images")
LANE_FILE = {0: "~/track_gt_lane1_demo.json", 1: "~/track_gt_lane0_demo.json"}  # 0=1차선중심선,1=2차선바깥
CMD = {0: "1차선 한바퀴 돌아", 1: "2차선 한바퀴 돌아"}
YAW_OFFSET = math.pi/2
WP_N, WP_STRIDE = 6, 5
SAVE_HZ = 10.0


def ego_wp(px, py, yaw, cl, i0):
    fwd = yaw - YAW_OFFSET; cf, sf = math.cos(fwd), math.sin(fwd); n = len(cl); out = []
    for k in range(1, WP_N+1):
        wx, wy = cl[(i0+k*WP_STRIDE) % n]; dx, dy = wx-px, wy-py
        out += [cf*dx+sf*dy, -sf*dx+cf*dy]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", type=int, default=0)
    ap.add_argument("--secs", type=float, default=280)
    ap.add_argument("--maxoff", type=float, default=3.0)   # 이보다 멀면 폐기(복구불가)
    ap.add_argument("--start-idx", type=int, default=150)
    a, _ = ap.parse_known_args()
    os.makedirs(IMG_DIR, exist_ok=True)
    cl = [(float(x), float(y)) for x, y in json.load(open(os.path.expanduser(LANE_FILE[a.lane])))["centerline_world"]]
    N = len(cl)
    rclpy.init(); n = Node("wp_dagger"); br = CvBridge()
    img = {"v": None}; pose = {"x": None, "y": None, "yaw": 0.0}
    be = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
    n.create_subscription(Image, "camera/image_raw",
                          lambda m: img.__setitem__("v", br.imgmsg_to_cv2(m, "bgr8")), be)
    def ocb(m):
        p = m.pose.pose.position; o = m.pose.pose.orientation
        pose["x"], pose["y"] = p.x, p.y
        pose["yaw"] = math.atan2(2*(o.w*o.z+o.x*o.y), 1-2*(o.y*o.y+o.z*o.z))
    n.create_subscription(Odometry, "/odom", ocb, be)
    pub = n.create_publisher(String, "vla/command",
                             QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                                        durability=QoSDurabilityPolicy.VOLATILE))
    # 출발 텔레포트 + lap 명령
    c = n.create_client(SetEntityState, "/gazebo/set_entity_state"); c.wait_for_service(timeout_sec=10)
    i0 = a.start_idx; x0, y0 = cl[i0]; nx, ny = cl[(i0+1) % N]; yaw = math.atan2(ny-y0, nx-x0)+math.pi/2
    r = SetEntityState.Request(); r.state.name = "ego_vehicle"
    r.state.pose.position.x = x0; r.state.pose.position.y = y0; r.state.pose.position.z = 0.05
    r.state.pose.orientation.z = math.sin(yaw/2); r.state.pose.orientation.w = math.cos(yaw/2); r.state.reference_frame = "world"
    rclpy.spin_until_future_complete(n, c.call_async(r), timeout_sec=2)
    time.sleep(0.6); pub.publish(String(data=CMD[a.lane])); time.sleep(0.4)

    csvp = os.path.join(HERE, "dataset", f"labels_wpLdag{a.lane}.csv")
    new = not os.path.exists(csvp)
    f = open(csvp, "a", newline="")
    if new:
        f.write("fname," + ",".join(f"{c}{k}" for k in range(WP_N) for c in ("ex", "ey")) + ",lane\n")
    last = 0.0; saved = 0; kept = 0; t0 = time.time()
    while time.time()-t0 < a.secs:
        rclpy.spin_once(n, timeout_sec=0.02)
        now = time.time()
        if now-last < 1.0/SAVE_HZ or img["v"] is None or pose["x"] is None:
            continue
        last = now; px, py, yw = pose["x"], pose["y"], pose["yaw"]
        i = min(range(N), key=lambda k: (cl[k][0]-px)**2+(cl[k][1]-py)**2)
        off = math.dist((px, py), cl[i])
        saved += 1
        if off > a.maxoff:        # 복구불가 과대이탈 폐기
            continue
        wps = ego_wp(px, py, yw, cl, i)
        if wps[0] < 0:            # 역방향 프레임 폐기
            continue
        fn = f"dag_L{a.lane}_{int(t0)}_{saved:06d}.jpg"
        cv2.imwrite(os.path.join(IMG_DIR, fn), img["v"])
        f.write(fn + "," + ",".join(f"{v:.3f}" for v in wps) + f",{a.lane}\n"); f.flush()
        kept += 1
        if kept % 50 == 0:
            n.get_logger().info(f"lane{a.lane}: 수집 {kept} (off={off:.2f}m)")
    f.close(); pub.publish(String(data="멈춰"))
    print(f"✓ DAgger lane{a.lane}: {kept}장 수집(폐기 {saved-kept}) → {csvp}")
    n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()

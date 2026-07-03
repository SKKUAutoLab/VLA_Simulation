#!/usr/bin/env python3
"""
B: 차선 변경(transition) WP 데이터 생성 — sim 텔레포트.
차선A 위(±오프셋)에 차를 두고, 미래 WP가 차선B로 부드럽게 합류(blend)하도록 라벨.
조건(lane_id): 2 = "2차선으로 변경"(1차선→2차선), 3 = "1차선으로 변경"(2차선→1차선).
저장: dataset/labels_wpLchg{2,3}.csv  (train_vla_lora 가 glob으로 흡수, lane=2/3)
주의: 끝나면 wp 이미지 재생성과 무관(chg_*.jpg 별도 이름).
"""
import os, csv, math, json, time
import cv2, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from gazebo_msgs.srv import SetEntityState

HERE = os.path.dirname(__file__); IMG_DIR = os.path.join(HERE, "dataset", "images")
LANE1 = os.path.expanduser("~/track_gt_lane1_demo.json")   # 1차선(중심)
LANE2 = os.path.expanduser("~/track_gt_lane0_demo.json")   # 2차선(바깥)
STEP = 4
LAT = [-0.4, 0.0, 0.4]         # 출발 차선 위 소오프셋(복구 겸용)
HEAD = [-10, 0, 10]
WP_N, WP_STRIDE = 6, 5
YAW_OFFSET = math.pi/2
Z = 0.05; SETTLE = 0.15
# 전이: A차선에서 시작해 ~WP_N*WP_STRIDE 구간에 걸쳐 B차선으로 합류
# blend alpha_k = k/WP_N (0=A, 1=B)


def load(p): return [(float(a), float(b)) for a, b in json.load(open(p))["centerline_world"]]
def nearest(cl, x, y): return min(range(len(cl)), key=lambda k: (cl[k][0]-x)**2+(cl[k][1]-y)**2)


def main():
    A_all = {0: load(LANE1), 1: load(LANE2)}   # 0=1차선,1=2차선
    rclpy.init(); n = Node("chg_gen"); br = CvBridge(); latest = {"img": None}
    n.create_subscription(Image, "camera/image_raw",
                          lambda m: latest.__setitem__("img", br.imgmsg_to_cv2(m, "bgr8")),
                          QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT))
    cli = n.create_client(SetEntityState, "/gazebo/set_entity_state"); cli.wait_for_service(timeout_sec=10)

    def teleport(x, y, yaw):
        r = SetEntityState.Request(); r.state.name = "ego_vehicle"
        r.state.pose.position.x = float(x); r.state.pose.position.y = float(y); r.state.pose.position.z = Z
        r.state.pose.orientation.z = math.sin(yaw/2); r.state.pose.orientation.w = math.cos(yaw/2); r.state.reference_frame = "world"
        rclpy.spin_until_future_complete(n, cli.call_async(r), timeout_sec=2)

    def grab():
        t0 = time.time()
        while time.time()-t0 < SETTLE: rclpy.spin_once(n, timeout_sec=0.02)
        return latest["img"]

    # (출발차선 srcL, 목적차선 dstL, 조건id)
    jobs = [(0, 1, 2), (1, 0, 3)]   # 1→2(id2), 2→1(id3)
    hdr = ["fname"] + [f"{c}{k}" for k in range(WP_N) for c in ("ex", "ey")] + ["lane"]
    for srcL, dstL, cid in jobs:
        A = A_all[srcL]; B = A_all[dstL]; nA = len(A); nB = len(B)
        f = open(os.path.join(HERE, "dataset", f"labels_wpLchg{cid}.csv"), "w", newline="")
        w = csv.writer(f); w.writerow(hdr); cnt = 0
        for i in range(0, nA, STEP):
            x0, y0 = A[i]; nx, ny = A[(i+1) % nA]; tang = math.atan2(ny-y0, nx-x0); normal = tang+math.pi/2
            jB = nearest(B, x0, y0)   # B에서 대응 지점
            for dl in LAT:
                px = x0+dl*math.cos(normal); py = y0+dl*math.sin(normal)
                for dh in HEAD:
                    yaw = tang+math.pi/2+math.radians(dh)
                    teleport(px, py, yaw); img = grab()
                    if img is None: continue
                    fwd = yaw-YAW_OFFSET; cf, sf = math.cos(fwd), math.sin(fwd); ego = []
                    for k in range(1, WP_N+1):
                        a = k/WP_N                              # 0→1 합류 비율
                        ax, ay = A[(i+k*WP_STRIDE) % nA]
                        bx, by = B[(jB+k*WP_STRIDE) % nB]
                        wx = (1-a)*ax + a*bx; wy = (1-a)*ay + a*by   # A→B 블렌드
                        dx, dy = wx-px, wy-py
                        ego += [cf*dx+sf*dy, -sf*dx+cf*dy]
                    fn = f"chg{cid}_{cnt:06d}.jpg"
                    cv2.imwrite(os.path.join(IMG_DIR, fn), img)
                    w.writerow([fn]+[f"{v:.3f}" for v in ego]+[cid]); cnt += 1
                    if cnt % 200 == 0: f.flush(); n.get_logger().info(f"chg{cid}: {cnt} (idx {i}/{nA})")
        f.close(); n.get_logger().info(f"✅ chg{cid} 완료: {cnt}")
    rclpy.shutdown()


if __name__ == "__main__":
    main()

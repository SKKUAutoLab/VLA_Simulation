#!/usr/bin/env python3
"""
DAgger 수집 — 모델이 주행하는 동안(on-policy) 카메라 + 실제 pose를 보고,
그 pose에서의 '정답 조향'(중앙선 복귀+추종, 기하)을 라벨로 저장.
모델이 실제로 가는 드리프트 상태(예: 코너 2m 이탈)에 정답을 달아 복구 학습.
사용: (모델 lap 주행 중) python3 collect_dagger.py --lane 0 --secs 160
저장: dataset/images/dag_*.jpg, dataset/labels_gtL{lane}_dag.csv
"""
import argparse, os, csv, math, time
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import lane_pursuit_expert as E

HERE = os.path.dirname(__file__)
IMG_DIR = os.path.join(HERE, "dataset", "images")
LANE_FILE = {0: os.path.expanduser("~/track_gt_manual.json"),
             1: os.path.expanduser("~/track_gt_lane2_centerline.json")}
SAVE_HZ = 8.0
LABEL_LOOKAHEAD = 2.5
MAX_OFF = 2.5   # 차선중앙 이탈 이 이상이면 버림(트랙밖 극단상태 제외)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", type=int, default=0)
    ap.add_argument("--secs", type=float, default=160)
    ap.add_argument("--maxoff", type=float, default=MAX_OFF)
    a, _ = ap.parse_known_args()
    maxoff = a.maxoff
    cl = [(float(x), float(y)) for x, y in __import__("json").load(open(LANE_FILE[a.lane]))["centerline_world"]]
    os.makedirs(IMG_DIR, exist_ok=True)
    rclpy.init(); n = Node("dagger"); br = CvBridge()
    pose = {"x": None, "y": None, "yaw": None}; img = {"v": None}
    def odom(m):
        p = m.pose.pose.position; o = m.pose.pose.orientation
        pose["x"], pose["y"] = p.x, p.y
        pose["yaw"] = math.atan2(2*(o.w*o.z+o.x*o.y), 1-2*(o.y*o.y+o.z*o.z))
    n.create_subscription(Odometry, "/odom", odom,
                          QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT))
    n.create_subscription(Image, "camera/image_raw",
                          lambda m: img.__setitem__("v", br.imgmsg_to_cv2(m, "bgr8")),
                          QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT))
    csvf = open(os.path.join(HERE, "dataset", f"labels_gtL{a.lane}_dag.csv"), "a", newline="")
    w = csv.writer(csvf)
    if os.path.getsize(csvf.name) == 0:
        w.writerow(["fname", "steering", "left_speed", "right_speed", "lane"])
    t0 = time.time(); last = 0; seq = 0; skipped = [0]
    while time.time() - t0 < a.secs and rclpy.ok():
        rclpy.spin_once(n, timeout_sec=0.02)
        now = time.time()
        if now - last < 1.0/SAVE_HZ or pose["x"] is None or img["v"] is None:
            continue
        last = now
        # 트랙밖 극단상태 필터: 차선중앙 이탈 > MAX_OFF면 버림
        ci = min(range(len(cl)), key=lambda k: (cl[k][0]-pose["x"])**2 + (cl[k][1]-pose["y"])**2)
        off = math.dist((pose["x"], pose["y"]), cl[ci])
        if off > maxoff:
            skipped[0] += 1
            continue
        st, sp, _ = E.pursuit_control(pose["x"], pose["y"], pose["yaw"], cl, lookahead=LABEL_LOOKAHEAD)
        fn = f"dag_L{a.lane}_{int(t0)}_{seq:05d}.jpg"
        cv2.imwrite(os.path.join(IMG_DIR, fn), img["v"])
        w.writerow([fn, st, sp, sp, a.lane]); csvf.flush(); seq += 1
        if seq % 50 == 0:
            n.get_logger().info(f"DAgger lane{a.lane}: {seq}장 (현 pose 정답 st={st})")
    csvf.close(); print(f"DAgger 완료: {seq}장 (필터로 버림 {skipped[0]}) → labels_gtL{a.lane}_dag.csv")
    n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()

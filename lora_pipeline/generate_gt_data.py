#!/usr/bin/env python3
"""
GT 데이터 생성 — 차를 차선 중앙선 포즈에 텔레포트하고 정답 조향을 라벨로 저장.
자율주행/노이즈 없이, 사용자가 찍어준 중앙선이 정답.
  - 각도 = 중앙선 접선(yaw=tangent+π/2, Prius 오프셋)
  - 강건성 위해 각 점에서 가로오프셋·heading 변형, 라벨은 그 포즈의
    '중앙선 복귀+추종' 기하 정답조향(pursuit_control).
저장: dataset/images/gt_*.jpg, dataset/labels_gtL0.csv / labels_gtL1.csv
"""
import os, csv, math, time
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from gazebo_msgs.srv import SetEntityState
import lane_pursuit_expert as E

HERE = os.path.dirname(__file__)
IMG_DIR = os.path.join(HERE, "dataset", "images")
LANES = {0: os.path.expanduser("~/track_gt_manual.json"),          # 1차선 (track_data TRACK_CENTERLINE)
         1: os.path.expanduser("~/track_gt_lane2_centerline.json")}  # 2차선 (infield쪽 2.5m, track_data LANE_SEPARATION)
STEP = 5                       # 중앙선 점(연속 traversal 근사)
LAT = [-0.5, 0.0, 0.5]         # 가로 오프셋(m): 0=중앙선, ±=복구 정답
HEAD = [-10, 0, 10]            # heading 변형(deg): 복구 정답
Z = 0.05
SETTLE = 0.06                  # 이동 후 렌더 대기[s]
LABEL_LOOKAHEAD = 2.5          # 라벨용 전방주시(↑ 곡률 예측 → 커브서 0 아닌 정답)


def yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw/2), math.cos(yaw/2))


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    rclpy.init()
    n = Node("gt_gen")
    br = CvBridge()
    latest = {"img": None}
    n.create_subscription(Image, "camera/image_raw",
                          lambda m: latest.__setitem__("img", br.imgmsg_to_cv2(m, "bgr8")),
                          QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT))
    cli = n.create_client(SetEntityState, "/gazebo/set_entity_state")
    cli.wait_for_service(timeout_sec=10)

    def teleport(x, y, yaw):
        req = SetEntityState.Request()
        req.state.name = "ego_vehicle"
        req.state.pose.position.x = float(x); req.state.pose.position.y = float(y)
        req.state.pose.position.z = Z
        q = yaw_to_quat(yaw)
        req.state.pose.orientation.x = q[0]; req.state.pose.orientation.y = q[1]
        req.state.pose.orientation.z = q[2]; req.state.pose.orientation.w = q[3]
        req.state.reference_frame = "world"
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(n, fut, timeout_sec=2.0)

    def grab():
        t0 = time.time()
        while time.time() - t0 < SETTLE:
            rclpy.spin_once(n, timeout_sec=0.02)
        return latest["img"]

    for lane, path in LANES.items():
        cl = [(float(a), float(b)) for a, b in __import__("json").load(open(path))["centerline_world"]]
        nP = len(cl)
        csvf = open(os.path.join(HERE, "dataset", f"labels_gtL{lane}.csv"), "w", newline="")
        w = csv.writer(csvf); w.writerow(["fname", "steering", "left_speed", "right_speed", "lane"])
        cnt = 0
        for i in range(0, nP, STEP):
            x0, y0 = cl[i]
            nx, ny = cl[(i+1) % nP]
            tang = math.atan2(ny-y0, nx-x0)
            normal = tang + math.pi/2
            for dl in LAT:
                px = x0 + dl*math.cos(normal); py = y0 + dl*math.sin(normal)
                for dh in HEAD:
                    yaw = tang + math.pi/2 + math.radians(dh)
                    teleport(px, py, yaw)
                    img = grab()
                    if img is None:
                        continue
                    # 정답 조향: 이 포즈에서 중앙선 복귀+추종 (기하, 곡률예측 lookahead)
                    st, sp, _ = E.pursuit_control(px, py, yaw, cl, lookahead=LABEL_LOOKAHEAD)
                    fn = f"gt_L{lane}_{cnt:06d}.jpg"
                    cv2.imwrite(os.path.join(IMG_DIR, fn), img)
                    w.writerow([fn, st, sp, sp, lane]); cnt += 1
                    if cnt % 200 == 0:
                        csvf.flush()
                        n.get_logger().info(f"lane{lane}: {cnt}장 (idx {i}/{nP})")
        csvf.close()
        n.get_logger().info(f"✅ lane{lane} 완료: {cnt}장")
    rclpy.shutdown()


if __name__ == "__main__":
    main()

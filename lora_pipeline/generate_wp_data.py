#!/usr/bin/env python3
"""
웨이포인트 GT 생성 — 차를 차선 중앙선 포즈(+오프셋)에 텔레포트하고,
'미래 중앙선 점 N개를 ego 프레임 좌표'로 라벨 저장. (조향 대신 경로 예측)
pure-pursuit가 추론 시 이 점들로 정확한 조향을 계산 → 과소조향 회피.
저장: dataset/images/wp_*.jpg, dataset/labels_wpL{lane}.csv
"""
import os, csv, math, time, json
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from gazebo_msgs.srv import SetEntityState

HERE = os.path.dirname(__file__)
IMG_DIR = os.path.join(HERE, "dataset", "images")
# 차선=사용자 주석과 0.2m내 일치하는 시연-적합 라인. 올바른 naming:
# lane0=1차선(중심선=inner=TRACK_CENTERLINE), lane1=2차선(바깥=outer)
LANES = {0: os.path.expanduser("~/track_gt_lane1_demo.json"),
         1: os.path.expanduser("~/track_gt_lane0_demo.json")}
# 기존 무접미사 데이터 = 역방향(사용자 기준). FWD=1 이면 그 반대편=정방향(차를 돌려세워 역순 WP) 생성.
FWD = os.environ.get("FWD") == "1"
SUF = "fwd" if FWD else ""
STEP = int(os.environ.get("WP_STEP", "3"))   # 조밀(복구 위치 다양성↑)
# 넓은 가로오프셋: off-center 복구 커버리지 대폭↑ (closed-loop robustness 핵심)
LAT = [-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9]
HEAD = [-15, 0, 15]            # heading 변형(deg)
Z = 0.05
SETTLE = 0.15                  # 텔레포트 후 신선 프레임 확보(저FPS 대비)
WP_N = 6                       # 미래 웨이포인트 수
WP_STRIDE = 5                  # 중앙선 인덱스 간격(~0.8m)
YAW_OFFSET = math.pi / 2


def yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw/2), math.cos(yaw/2))


def ego_waypoints(px, py, yaw, cl, i0):
    """포즈(px,py,yaw)에서 i0부터 미래 N점을 ego(x=전방,y=좌) 좌표로.
    역방향(REV)이면 인덱스 역순으로 미래점을 잡음(차가 반대로 진행)."""
    fwd = yaw - YAW_OFFSET
    cf, sf = math.cos(fwd), math.sin(fwd)
    n = len(cl); out = []
    sgn = -1 if FWD else 1
    for k in range(1, WP_N+1):
        wx, wy = cl[(i0 + sgn*k*WP_STRIDE) % n]
        dx, dy = wx-px, wy-py
        ex = cf*dx + sf*dy      # 전방
        ey = -sf*dx + cf*dy     # 좌
        out += [ex, ey]
    return out


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    rclpy.init(); n = Node("wp_gen"); br = CvBridge()
    latest = {"img": None}
    n.create_subscription(Image, "camera/image_raw",
                          lambda m: latest.__setitem__("img", br.imgmsg_to_cv2(m, "bgr8")),
                          QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT))
    cli = n.create_client(SetEntityState, "/gazebo/set_entity_state")
    cli.wait_for_service(timeout_sec=10)

    def teleport(x, y, yaw):
        req = SetEntityState.Request(); req.state.name = "ego_vehicle"
        req.state.pose.position.x = float(x); req.state.pose.position.y = float(y); req.state.pose.position.z = Z
        q = yaw_to_quat(yaw)
        req.state.pose.orientation.z = q[2]; req.state.pose.orientation.w = q[3]
        req.state.reference_frame = "world"
        rclpy.spin_until_future_complete(n, cli.call_async(req), timeout_sec=2.0)

    def grab():
        t0 = time.time()
        while time.time()-t0 < SETTLE:
            rclpy.spin_once(n, timeout_sec=0.02)
        return latest["img"]

    hdr = ["fname"] + [f"{c}{k}" for k in range(WP_N) for c in ("ex", "ey")] + ["lane"]
    only = os.environ.get("GEN_LANE")   # "0"/"1"면 그 차선만 생성
    lanes = {int(only): LANES[int(only)]} if only in ("0", "1") else LANES
    for lane, path in lanes.items():
        cl = [(float(a), float(b)) for a, b in json.load(open(path))["centerline_world"]]
        nP = len(cl)
        f = open(os.path.join(HERE, "dataset", f"labels_wpL{lane}{SUF}.csv"), "w", newline="")
        w = csv.writer(f); w.writerow(hdr); cnt = 0
        for i in range(0, nP, STEP):
            x0, y0 = cl[i]; nx, ny = cl[(i+1) % nP]
            tang = math.atan2(ny-y0, nx-x0); normal = tang + math.pi/2
            for dl in LAT:
                px = x0 + dl*math.cos(normal); py = y0 + dl*math.sin(normal)
                for dh in HEAD:
                    yaw = tang + math.pi/2 + (math.pi if FWD else 0.0) + math.radians(dh)
                    teleport(px, py, yaw); img = grab()
                    if img is None:
                        continue
                    wps = ego_waypoints(px, py, yaw, cl, i)
                    fn = f"wp_L{lane}{SUF}_{cnt:06d}.jpg"
                    cv2.imwrite(os.path.join(IMG_DIR, fn), img)
                    w.writerow([fn] + [f"{v:.3f}" for v in wps] + [lane]); cnt += 1
                    if cnt % 200 == 0:
                        f.flush(); n.get_logger().info(f"lane{lane}: {cnt} (idx {i}/{nP})")
        f.close(); n.get_logger().info(f"✅ lane{lane} 완료: {cnt}장")
    rclpy.shutdown()


if __name__ == "__main__":
    main()

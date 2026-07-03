#!/usr/bin/env python3
"""
특권(privileged) expert — 차선중앙 GT(track_gt_manual.json: centerline_world 720pts)를
/odom 좌표로 pure-pursuit 추종해 도로를 자동주행한다. topic_control_signal 발행.
perception 불필요. LoRA 데모 자동수집용 (사람 운전 대체).

조향 법칙은 검증된 brain 규약 재사용:
  car_forward = yaw - π/2,  heading_err = atan2(target-car) - car_forward,
  steering = clip(-heading_err * GAIN, -7, 7)   (steering>0 → 우회전, sender STEER=-1)

실행:
  ros2 launch simulation_pkg teleop_sim.launch.py     # 차+카메라+sender (planner 없음)
  python3 lora_pipeline/lane_pursuit_expert.py        # 본 expert
  python3 lora_pipeline/collect_demos_node.py         # 수집
오프라인 검증:
  python3 lora_pipeline/lane_pursuit_expert.py --offline
"""
import os, json, math, argparse, random

GT_PATH = os.path.expanduser("~/track_gt_manual.json")
CONTROL_TOPIC = "topic_control_signal"
LABEL_TOPIC   = "vla/expert_label"   # 노이즈 없는 깨끗한 조향(라벨용)
ODOM_TOPIC = "/odom"

# 실측: gain 4.5/lookahead 1.5면 차가 자기 차선(중심선 GT는 inner라인 기준
# ~2.6m 옆)에 도로 위로 정상 주행. 과한 gain은 오히려 진동/정체 유발.
LOOKAHEAD   = 1.5     # [m] 전방주시 거리
GAIN        = 5.0     # heading_err(rad) → steering
BASE_SPEED  = 36      # 직선 속도 (left/right_speed 단위)
MIN_SPEED   = 22      # 급커브 최소 속도
FIXED_DIR   = 1       # 루프 진행방향 (centerline index 증가=+1). flip 방지용 고정
YAW_OFFSET  = math.pi / 2


def load_centerline(path=GT_PATH):
    d = json.load(open(path))
    return [(float(x), float(y)) for x, y in d["centerline_world"]]


def normalize_angle(a):
    while a > math.pi:  a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


def pursuit_control(x, y, yaw, cl, lookahead=LOOKAHEAD, gain=GAIN,
                    base_speed=BASE_SPEED):
    """현재 pose → (steering, speed, dbg). cl=centerline 리스트."""
    n = len(cl)
    # 최근접 인덱스
    i0 = min(range(n), key=lambda i: (cl[i][0]-x)**2 + (cl[i][1]-y)**2)
    car_forward = yaw - YAW_OFFSET
    # 진행방향 고정(+index). 매 틱 차 헤딩으로 재계산하면 코너에서 차가 경로와
    # 수직이 될 때 부호가 뒤집혀 진동함(실측 버그) → 루프 정방향(+1)으로 고정.
    direction = FIXED_DIR
    # lookahead 만큼 경로 따라 전진
    idx, acc = i0, 0.0
    for _ in range(n):
        nxt = (idx + direction) % n
        acc += math.dist(cl[idx], cl[nxt])
        idx = nxt
        if acc >= lookahead:
            break
    tx, ty = cl[idx]
    target_heading = math.atan2(ty - y, tx - x)
    he = normalize_angle(target_heading - car_forward)
    steering = max(-7, min(7, -he * gain))
    sp = base_speed if abs(steering) < 3 else max(MIN_SPEED,
                                                  base_speed - 4*abs(steering))
    return int(round(steering)), int(sp), {"i0": i0, "target": (tx, ty),
                                           "he_deg": math.degrees(he),
                                           "dir": direction}


# ─── 오프라인 키네마틱 검증 ──────────────────────────────────────────────
def offline_test():
    cl = load_centerline()
    n = len(cl)
    print(f"centerline {n}pts. 오프라인 추종 테스트.")
    # sender: angular.z = -steering*0.0923 (rad/s), 모션방향 = yaw - π/2
    YAW_RATE_PER_STEER = 0.6458 / 7      # ≈0.0923
    VMAX = 3.0                            # m/s @ speed=255 (대략)
    dt = 0.05

    for label, (sx, sy, syaw) in {
        "스폰(-2.55,-22.71) yaw=π": (-2.55, -22.71, math.pi),
        "경로위 idx0 정렬":          (cl[0][0], cl[0][1], None),
    }.items():
        x, y = sx, sy
        if syaw is None:
            # idx0 접선 방향으로 정렬
            tx, ty = cl[1][0]-cl[0][0], cl[1][1]-cl[0][1]
            yaw = math.atan2(ty, tx) + YAW_OFFSET
        else:
            yaw = syaw
        max_off = 0.0
        visited = set()
        steps = int(180 / dt)   # 최대 180초
        for k in range(steps):
            st, sp, dbg = pursuit_control(x, y, yaw, cl)
            visited.add(dbg["i0"] // 10)
            # 경로 이탈 측정
            off = math.dist((x, y), cl[dbg["i0"]])
            max_off = max(max_off, off)
            # 키네마틱 적분
            yaw_rate = -st * YAW_RATE_PER_STEER
            v = (sp / 255.0) * VMAX
            mdir = yaw - YAW_OFFSET
            x += v * math.cos(mdir) * dt
            y += v * math.sin(mdir) * dt
            yaw += yaw_rate * dt
        cov = 100 * len(visited) / (n/10)
        ok = "✅" if (max_off < 1.5 and cov > 80) else "⚠️"
        print(f"  {ok} {label}: 경로커버 {cov:.0f}%  최대이탈 {max_off:.2f}m")
    print("커버≈100%·이탈<1.5m면 한 바퀴 정상 추종.")


# ─── ROS 노드 ──────────────────────────────────────────────────────────
def ros_main(centerline_path=GT_PATH, direction=1, noise=0.0):
    import rclpy
    globals()["FIXED_DIR"] = direction   # 진행방향 설정(±1)
    from rclpy.node import Node
    from rclpy.qos import (QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy,
                           QoSDurabilityPolicy)
    from nav_msgs.msg import Odometry
    from interfaces_pkg.msg import MotionCommand

    def quat_to_yaw(qx, qy, qz, qw):
        return math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))

    class ExpertNode(Node):
        def __init__(self):
            super().__init__("lane_pursuit_expert")
            self.cl = load_centerline(centerline_path)
            self.x = self.y = self.yaw = None
            qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                             history=QoSHistoryPolicy.KEEP_LAST,
                             durability=QoSDurabilityPolicy.VOLATILE, depth=1)
            self.pub = self.create_publisher(MotionCommand, CONTROL_TOPIC, qos)
            self.label_pub = self.create_publisher(MotionCommand, LABEL_TOPIC, qos)
            self.noise = noise
            self.create_subscription(
                Odometry, ODOM_TOPIC, self._odom,
                QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                           history=QoSHistoryPolicy.KEEP_LAST,
                           durability=QoSDurabilityPolicy.VOLATILE))
            self.create_timer(0.05, self._tick)   # 20Hz
            self.get_logger().info(f"expert ready. centerline {len(self.cl)}pts")

        def _odom(self, msg):
            p = msg.pose.pose.position; o = msg.pose.pose.orientation
            self.x, self.y = p.x, p.y
            self.yaw = quat_to_yaw(o.x, o.y, o.z, o.w)

        def _tick(self):
            if self.x is None:
                return
            st, sp, dbg = pursuit_control(self.x, self.y, self.yaw, self.cl)
            # 깨끗한 조향 = 라벨(복구 정답). 별도 토픽으로 발행.
            lm = MotionCommand(); lm.steering = st; lm.left_speed = sp; lm.right_speed = sp
            self.label_pub.publish(lm)
            # 실제 구동엔 노이즈 주입 → 차가 차선서 흔들림(복구 상태 생성, DART)
            drive_st = st
            if self.noise > 0:
                drive_st = int(max(-7, min(7, round(st + random.gauss(0, self.noise)))))
            m = MotionCommand(); m.steering = drive_st; m.left_speed = sp; m.right_speed = sp
            self.pub.publish(m)

    rclpy.init()
    node = ExpertNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="키네마틱 추종 검증")
    ap.add_argument("--centerline", default=GT_PATH,
                    help="추종할 중심선 json (기본 inner; outer는 ~/track_gt_outer_centerline.json)")
    ap.add_argument("--dir", type=int, default=1, help="루프 진행방향 +1/-1 (양방향 수집)")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="구동 조향 노이즈 std (복구데이터용, 예 2.5). 라벨은 깨끗")
    args, _ = ap.parse_known_args()
    if args.offline:
        offline_test()
    else:
        ros_main(args.centerline, args.dir, args.noise)


if __name__ == "__main__":
    main()

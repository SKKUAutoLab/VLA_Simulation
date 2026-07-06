#!/usr/bin/env python3
"""차도(2차선 centerline) 위 직선구간에 장애물차량 스폰/삭제 — GUI 토글용.
사용: python3 spawn_lane_obs.py on   |   python3 spawn_lane_obs.py off
회피 데모용으로 2차선 위에 스폰(1차선은 비워둠).
"""
import os, sys, math, json, re
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity, DeleteEntity

LANE = os.path.expanduser("~/track_gt_lane0_demo.json")        # 2차선(바깥)
HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
MODEL = os.path.join(WS, "src/simulation_pkg/models/prius_hybrid_ob1/model.sdf")
# 차도 직선구간(2차선)에 배치 — 추월/회피 데모는 소수가 적합(많으면 양쪽 막혀 정지).
IDXS_ALL = [480, 360, 700, 140]   # 상단·우측·하단·좌측 직선
OBS_N = int(os.environ.get("OBS_N", "1"))          # 기본 1대(추월 데모). 여러 대 원하면 OBS_N=4
IDXS = IDXS_ALL[:max(1, min(OBS_N, len(IDXS_ALL)))]
# 삭제는 항상 4개 슬롯 전부 대상(과거 많이 스폰된 것까지 정리)
NAMES = [f"laneobs{i+1}" for i in range(len(IDXS))]
ALL_NAMES = [f"laneobs{i+1}" for i in range(len(IDXS_ALL))]


def _delete_all(n):
    """기존 장애물 삭제 — 재토글 시 중복 스폰 방지(멱등). 슬롯 4개 전부 정리."""
    cli = n.create_client(DeleteEntity, "/delete_entity"); cli.wait_for_service(timeout_sec=10)
    for nm in ALL_NAMES:
        req = DeleteEntity.Request(); req.name = nm
        rclpy.spin_until_future_complete(n, cli.call_async(req), timeout_sec=8)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "on"
    rclpy.init()
    n = Node("spawn_lane_obs")
    # 스폰이든 삭제든 항상 먼저 기존 것을 제거(멱등) → 재토글해도 중복/이름충돌 없음.
    _delete_all(n)
    if mode == "off":
        print(f"삭제: {NAMES}")
    else:
        cl = [(float(a), float(b)) for a, b in json.load(open(LANE))["centerline_world"]]
        N = len(cl)
        sdf = open(MODEL).read()
        # 정적 장애물은 ROS 플러그인 불필요. 여러 대가 같은 노드이름(/ob1, ackermann/joint_state)으로
        # 충돌하면 gzserver가 세그폴트(exit -11)로 죽는다 → 모든 <plugin> 블록 제거 후 스폰.
        sdf = re.sub(r"<plugin\b.*?</plugin>", "", sdf, flags=re.DOTALL)
        cli = n.create_client(SpawnEntity, "/spawn_entity"); cli.wait_for_service(timeout_sec=10)
        for nm, idx in zip(NAMES, IDXS):
            x, y = cl[idx]; nx, ny = cl[(idx + 1) % N]
            yaw = math.atan2(ny - y, nx - x) + math.pi / 2     # 차선 진행방향(차량 정렬)
            req = SpawnEntity.Request(); req.name = nm; req.xml = sdf
            req.initial_pose.position.x = x; req.initial_pose.position.y = y; req.initial_pose.position.z = 0.1
            req.initial_pose.orientation.z = math.sin(yaw / 2); req.initial_pose.orientation.w = math.cos(yaw / 2)
            rclpy.spin_until_future_complete(n, cli.call_async(req), timeout_sec=10)
        print(f"차도(2차선) 스폰: {list(zip(NAMES, IDXS))}")
    n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()

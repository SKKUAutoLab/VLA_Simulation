#!/usr/bin/env python3
"""차도(2차선 centerline) 위 직선구간에 장애물차량 스폰/삭제 — GUI 토글용.
사용: python3 spawn_lane_obs.py on   |   python3 spawn_lane_obs.py off
회피 데모용으로 2차선 위에 스폰(1차선은 비워둠).
"""
import os, sys, math, json
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity, DeleteEntity

LANE = os.path.expanduser("~/track_gt_lane0_demo.json")        # 2차선(바깥)
HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
MODEL = os.path.join(WS, "src/simulation_pkg/models/prius_hybrid_ob1/model.sdf")
# 차도 직선구간(2차선)에 띄엄띄엄 — 회피 여유 위해 직선 위주
IDXS = [360, 480, 700, 140]   # 상단·우측·하단·좌측 직선
NAMES = [f"laneobs{i+1}" for i in range(len(IDXS))]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "on"
    rclpy.init()
    n = Node("spawn_lane_obs")
    if mode == "off":
        cli = n.create_client(DeleteEntity, "/delete_entity"); cli.wait_for_service(timeout_sec=10)
        for nm in NAMES:
            req = DeleteEntity.Request(); req.name = nm
            rclpy.spin_until_future_complete(n, cli.call_async(req), timeout_sec=8)
        print(f"삭제: {NAMES}")
    else:
        cl = [(float(a), float(b)) for a, b in json.load(open(LANE))["centerline_world"]]
        N = len(cl)
        sdf = open(MODEL).read()
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

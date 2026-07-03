#!/usr/bin/env python3
"""
VLA Agent 시뮬레이션 런치 파일 (MCP 스타일 순수 VLA 직접 제어)

실행:
    ros2 launch simulation_pkg vla_agent_sim.launch.py

노드 구성:
    1. Gazebo (track.world)
    2. ego car 생성
    3. VLA Agent (Qwen3-VL이 tool 호출로 차량 직접 제어, 기하 제어 없음)
    4. Sim Sender (MotionCommand → Gazebo 제어)
    + lidar_processor (안전 정지용)

명령 입력 (별도 터미널):
    ros2 run qwen_vl_pkg vla_cmd_node
    또는: ros2 topic pub /vla/goal_cmd std_msgs/msg/String "data: 'P2에 주차해'" --once

전제: ~/track_features.json 이 존재해야 함 (gt_annotator 피처 저장본).
"""

import os
import subprocess
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


nvidia_env = {
    "__NV_PRIME_RENDER_OFFLOAD":  "1",
    "__GLX_VENDOR_LIBRARY_NAME":  "nvidia",
    "__VK_LAYER_NV_optimus":      "NVIDIA_only",
}


def generate_launch_description():
    subprocess.run(["killall", "gzserver"], capture_output=True)
    subprocess.run(["killall", "gzclient"], capture_output=True)

    pkg = get_package_share_directory("simulation_pkg")
    world = os.path.join(pkg, "worlds", "track.world")

    return LaunchDescription([
        ExecuteProcess(
            cmd=["gazebo", world, "-s", "libgazebo_ros_factory.so"],
            additional_env=nvidia_env,
            output="screen",
        ),
        Node(package="simulation_pkg", executable="load_ego_car_node",
             output="screen"),
        # ── 장애물 차량 + 신호등 스폰 ────────────────────────────────────
        Node(package="simulation_pkg", executable="load_obstable_car_node",
             output="screen"),
        Node(package="simulation_pkg", executable="load_traffic_light_node",
             output="screen"),
        Node(package="lidar_perception_pkg", executable="lidar_processor_node",
             output="screen"),
        # ── VLA Agent (순수 VLM 직접 제어) ───────────────────────────────
        Node(package="qwen_vl_pkg", executable="vla_agent_node",
             output="screen"),
        Node(package="simulation_pkg", executable="sim_simulation_sender_node",
             output="screen"),
    ])

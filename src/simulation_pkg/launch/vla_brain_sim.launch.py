#!/usr/bin/env python3
"""
VLA Brain 시뮬레이션 런치 파일

실행:
    ros2 launch simulation_pkg vla_brain_sim.launch.py

노드 구성:
    1. Gazebo (track.world)
    2. ego car 생성
    3. VLA Brain (Qwen3-VL + 기하학 내비게이션 + 장애물 감속)
    4. Sim Sender (MotionCommand → Gazebo 제어)
    [선택] lidar_processor, lidar_obstacle_detector 자동 포함

명령 입력 (별도 터미널):
    ros2 run qwen_vl_pkg vla_cmd_node
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

        # ── Gazebo ──────────────────────────────────────────────────────
        ExecuteProcess(
            cmd=["gazebo", world, "-s", "libgazebo_ros_factory.so"],
            additional_env=nvidia_env,
            output="screen",
        ),

        # ── Ego car 생성 ─────────────────────────────────────────────────
        Node(
            package="simulation_pkg",
            executable="load_ego_car_node",
            output="screen",
        ),

        # ── LiDAR 전처리 ─────────────────────────────────────────────────
        Node(
            package="lidar_perception_pkg",
            executable="lidar_processor_node",
            output="screen",
        ),

        # ── VLA Brain ────────────────────────────────────────────────────
        # 목표 명령: ros2 run qwen_vl_pkg vla_cmd_node (별도 터미널)
        Node(
            package="qwen_vl_pkg",
            executable="vla_brain_node",
            output="screen",
        ),

        # ── Simulation Sender ────────────────────────────────────────────
        Node(
            package="simulation_pkg",
            executable="sim_simulation_sender_node",
            output="screen",
        ),

    ])

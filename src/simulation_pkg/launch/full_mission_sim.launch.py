#!/usr/bin/env python3
"""
full_mission_sim.launch.py
==========================
미션 시뮬레이션 전체 스택 런치 파일

포함 노드:
  [시뮬레이션]
    - Gazebo (track.world)
    - load_ego_car_node      : 자차 스폰
    - load_obstable_car_node : 장애물 차량 스폰
    - load_traffic_light_node: 신호등 스폰

  [인지]
    - yolov8_node            : 객체 탐지 (차선, 신호등, 차량)
    - lane_info_extractor_node
    - lidar_processor_node   : LiDAR 전처리
    - lidar_obstacle_detector: 장애물 감지

  [판단]
    - motion_planner_node    : 기존 모션 플래너 (차선 추종)
    - task_manager_node      : 미션 태스크 관리 (신규)
    - vla_brain_node         : VLA 기반 자율주행 브레인 (선택)

  [시각화]
    - path_visualizer_node   : 경로 시각화
    - mission_gui_node       : PySide6 미션 제어 GUI (신규)

사용법:
  # 기본 실행 (GUI + 태스크 매니저 + 기존 스택)
  ros2 launch simulation_pkg full_mission_sim.launch.py

  # VLA 브레인 없이 실행
  ros2 launch simulation_pkg full_mission_sim.launch.py use_vla:=false

  # GUI 없이 실행 (헤드리스)
  ros2 launch simulation_pkg full_mission_sim.launch.py use_gui:=false
"""

import os
import subprocess
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import (ExecuteProcess, DeclareLaunchArgument,
                             OpaqueFunction, GroupAction)
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from ament_index_python.packages import get_package_share_directory

# NVIDIA GPU 환경변수
nvidia_env = {
    '__NV_PRIME_RENDER_OFFLOAD': '1',
    '__GLX_VENDOR_LIBRARY_NAME': 'nvidia',
    '__VK_LAYER_NV_optimus': 'NVIDIA_only',
}


def generate_launch_description():

    # 기존 Gazebo 프로세스 종료
    subprocess.run(['killall', 'gzserver'], capture_output=True)
    subprocess.run(['killall', 'gzclient'], capture_output=True)

    package_dir = get_package_share_directory('simulation_pkg')
    world_file  = os.path.join(package_dir, 'worlds', 'track.world')

    # ── 런치 인수 ──────────────────────────────────────────────────────
    use_vla_arg = DeclareLaunchArgument(
        'use_vla', default_value='true',
        description='VLA 브레인 노드 사용 여부 (true/false)'
    )
    use_gui_arg = DeclareLaunchArgument(
        'use_gui', default_value='true',
        description='PySide6 GUI 사용 여부 (true/false)'
    )

    use_vla = LaunchConfiguration('use_vla')
    use_gui = LaunchConfiguration('use_gui')

    return LaunchDescription([

        use_vla_arg,
        use_gui_arg,

        # ── 1. Gazebo 시뮬레이터 ───────────────────────────────────────
        ExecuteProcess(
            cmd=['gazebo', world_file, '-s', 'libgazebo_ros_factory.so'],
            additional_env=nvidia_env,
            output='screen',
        ),

        # ── 2. 시뮬레이션 초기화 ───────────────────────────────────────
        Node(
            package='simulation_pkg',
            executable='load_ego_car_node',
            output='screen',
        ),
        Node(
            package='simulation_pkg',
            executable='load_obstable_car_node',
            output='screen',
        ),
        Node(
            package='simulation_pkg',
            executable='load_traffic_light_node',
            output='screen',
        ),

        # ── 3. 인지 스택 ──────────────────────────────────────────────
        Node(
            package='camera_perception_pkg',
            executable='yolov8_node',
            output='screen',
        ),
        Node(
            package='camera_perception_pkg',
            executable='lane_info_extractor_node',
            output='screen',
        ),
        # LiDAR 처리 (simulation_pkg 내 sim 버전 사용)
        Node(
            package='simulation_pkg',
            executable='sim_lidar_processor_node',
            output='screen',
        ),
        Node(
            package='simulation_pkg',
            executable='sim_lidar_obstacle_detector_node',
            output='screen',
        ),

        # ── 4. 미션 태스크 매니저 (신규) ──────────────────────────────
        Node(
            package='mission_control_pkg',
            executable='task_manager_node',
            output='screen',
            name='task_manager_node',
        ),

        # ── 5. VLA 브레인 (선택) ────────────────────────────────────
        Node(
            package='qwen_vl_pkg',
            executable='vla_brain_node',
            output='screen',
            condition=IfCondition(use_vla),
        ),

        # ── 6. 기존 판단 스택 (VLA 없을 때) ─────────────────────────
        Node(
            package='decision_making_pkg',
            executable='path_planner_node',
            output='screen',
            condition=UnlessCondition(use_vla),
        ),
        Node(
            package='decision_making_pkg',
            executable='motion_planner_node',
            output='screen',
            condition=UnlessCondition(use_vla),
        ),

        # ── 7. 제어 송신 ──────────────────────────────────────────────
        Node(
            package='simulation_pkg',
            executable='sim_simulation_sender_node',
            output='screen',
        ),

        # ── 8. 시각화 ─────────────────────────────────────────────────
        Node(
            package='debug_pkg',
            executable='yolov8_visualizer_node',
            output='screen',
        ),
        Node(
            package='debug_pkg',
            executable='path_visualizer_node',
            output='screen',
        ),

        # ── 9. PySide6 GUI (선택) ─────────────────────────────────────
        Node(
            package='gui_pkg',
            executable='mission_gui_node',
            output='screen',
            condition=IfCondition(use_gui),
        ),

    ])
